import atexit
import logging
import shutil
import sys
import time

import backend
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from agent.mcts_agent import MCTSAgent as Agent
from interpreter.interpreter_parallel import Interpreter
from search.journal import Journal
from search.node import Node
from omegaconf import OmegaConf
from rich.columns import Columns
from rich.console import Group
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.markdown import Markdown
from rich.status import Status
from rich.tree import Tree
from utils.config_mcts import load_task_desc, prep_agent_workspace, save_run, load_cfg
from utils.serialize import load_json
from utils.event_stream import configure_event_sink, emit_event

class VerboseFilter(logging.Filter):
    """
    Filter (remove) logs that have verbose attribute set to True
    """

    def filter(self, record):
        return not (hasattr(record, "verbose") and record.verbose)


def journal_to_rich_tree(journal: Journal):
    best_node = journal.get_best_node()

    def append_rec(node: Node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""

            if node is best_node:
                s = f"[{style}green]● {node.metric.value:.3f} (best)"
            else:
                s = f"[{style}green]● {node.metric.value:.3f}"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree


def journal_to_string_tree(journal: Journal) -> str:
    best_node = journal.get_best_node()
    tree_str = "Solution tree\n"

    def append_rec(node: Node, level: int):
        nonlocal tree_str
        indent = "  " * level
        if node.is_buggy:
            s = f"{indent}◍ bug (ID: {node.id})\n"
        else:
            # support for multiple markers; atm only "best" is supported
            markers = []
            if node is best_node:
                markers.append("best")
            marker_str = " & ".join(markers)
            if marker_str and node.metric.value:
                s = f"{indent}● {node.metric.value:.3f} ({marker_str}) (ID: {node.id})\n"
            else:
                s = f"{indent}● {node.metric.value:.3f} (ID: {node.id})\n"
        tree_str += s
        for child in node.children:
            append_rec(child, level + 1)

    for n in journal.draft_nodes:
        append_rec(n, 0)

    return tree_str


def _repair_mcts_journal_state(journal: Journal):
    if len(journal) == 0:
        return journal
    id2node = {n.id: n for n in journal.nodes}
    for n in journal.nodes:
        if getattr(n, "children", None) is None:
            n.children = set()
    for n in journal.nodes:
        p = getattr(n, "parent", None)
        if p is None:
            continue
        if p.id in id2node:
            id2node[p.id].children.add(n)
    return journal


def run():
    cfg = load_cfg()
    configure_event_sink(cfg.log_dir / "event_stream.jsonl")
    emit_event("pipeline", "RUN_STARTED", exp_name=cfg.exp_name)
    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()), format=log_format, handlers=[]
    )
    # dont want info logs from httpx
    httpx_logger: logging.Logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)

    logger = logging.getLogger("ml-master")
    # save logs to files as well, using same format
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # we'll have a normal log file and verbose log file. Only normal to console
    file_handler = logging.FileHandler(cfg.log_dir / "ml-master.log")
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.addFilter(VerboseFilter())

    verbose_file_handler = logging.FileHandler(cfg.log_dir / "ml-master.verbose.log")
    verbose_file_handler.setFormatter(logging.Formatter(log_format))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.addFilter(VerboseFilter())

    logger.addHandler(file_handler)
    logger.addHandler(verbose_file_handler)
    logger.addHandler(console_handler)

    logger.info(f'Starting run "{cfg.exp_name}"')
    emit_event("pipeline", "CONFIG_READY", log_dir=str(cfg.log_dir), workspace_dir=str(cfg.workspace_dir))

    task_desc = load_task_desc(cfg)
    task_desc_str = backend.compile_prompt_to_md(task_desc)

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)
    emit_event("pipeline", "WORKSPACE_READY")

    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    if cfg.agent.steerable_reasoning == True:
        logger.warning("Steerable reasoning is enabled, please make sure your open sourced model api support `client.compeletion.create()`, otherwise the process may fail")
        if "gpt" in cfg.agent.code.model or "gemini" in cfg.agent.code.model or "claude" in cfg.agent.code.model:
            logger.warning("Steerable reasoning does not support close sourced models, please set steerable reasoning to false")
            raise ValueError("Steerable reasoning does not support close sourced models, please set steerable reasoning to false")
    
    if cfg.agent.check_format == True:
        logger.warning("Check format is enabled, please make sure you have launched the server, or this step will be skipped")


    atexit.register(cleanup)

    resume_path = cfg.log_dir / "journal.json"
    if resume_path.exists():
        try:
            journal = load_json(resume_path, Journal)
            journal = _repair_mcts_journal_state(journal)
            logger.info(f"Resuming from existing journal: {resume_path}")
            emit_event("pipeline", "RUN_RESUMED", journal_path=str(resume_path), nodes=len(journal))
        except Exception as e:
            logger.warning(f"Failed to load existing journal, starting fresh: {e}")
            journal = Journal()
    else:
        journal = Journal()
    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
    )
    emit_event("agent.mcts", "CREATED", steps=cfg.agent.steps, parallel_search_num=cfg.agent.search.parallel_search_num)

    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec), cfg=cfg  # type: ignore
    )

    global_step = len(journal)
    prog = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )
    status = Status("[green]Generating code...")
    prog.add_task("Progress:", total=cfg.agent.steps, completed=global_step)

    def exec_callback(*args, **kwargs):
        status.update("[magenta]Executing code...")
        res = interpreter.run(*args, **kwargs)
        status.update("[green]Generating code...")
        return res

    def step_task(node=None):
        if node:
            logger.info(f"[step_task] Processing node: {node.id}")
            emit_event("agent.mcts", "STEP_STARTED", node_id=node.id, node_stage=node.stage)
        else:
            logger.info(f"[step_task] Processing virtual root node.")
            emit_event("agent.mcts", "STEP_STARTED", node_id="root", node_stage="root")
        return agent.step(exec_callback=exec_callback, node=node)
    
    max_workers = cfg.agent.search.parallel_search_num
    total_steps = cfg.agent.steps
    global_deadline = time.time() + max(1, int(cfg.agent.time_limit))
    timed_out = False
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = set()
    completed = 0
    lock = threading.Lock()
    try:
        futures = {executor.submit(step_task) for _ in range(min(max_workers, total_steps))}
        while completed <= total_steps:
            now = time.time()
            if now >= global_deadline:
                timed_out = True
                logger.warning(
                    f"Global time limit reached ({cfg.agent.time_limit}s). "
                    "Stopping MCTS expansion and finalizing current artifacts."
                )
                emit_event(
                    "pipeline",
                    "RUN_TIMEOUT",
                    completed=int(completed),
                    total_steps=int(total_steps),
                    time_limit_secs=int(cfg.agent.time_limit),
                )
                break

            wait_timeout = max(0.1, min(2.0, global_deadline - now))
            done, _ = wait(futures, timeout=wait_timeout, return_when=FIRST_COMPLETED)
            if not done:
                continue

            for fut in done:
                futures.remove(fut)
                try:
                    cur_node = fut.result()
                    logger.info(f"current node count is {completed}, current node.id is {cur_node.id}")
                    emit_event("agent.mcts", "STEP_COMPLETED", next_node_id=cur_node.id, next_node_stage=cur_node.stage)
                except Exception as e:
                    logger.exception(f"Exception during step_task execution: {e}")
                    emit_event("agent.mcts", "STEP_FAILED", error=str(e))
                    cur_node = None

                with lock:
                    save_run(cfg, journal)
                    completed = len(journal)-1. # Exclude virtual node
                    emit_event("pipeline", "JOURNAL_SAVED", completed=int(completed), total_steps=int(total_steps))
                    if completed == total_steps:
                        logger.info(journal_to_string_tree(journal))
                        emit_event("pipeline", "RUN_COMPLETED", completed=int(completed))

                # No further expansion after deadline.
                if time.time() >= global_deadline:
                    timed_out = True
                    emit_event(
                        "pipeline",
                        "RUN_TIMEOUT",
                        completed=int(completed),
                        total_steps=int(total_steps),
                        time_limit_secs=int(cfg.agent.time_limit),
                    )
                    break

                if completed + len(futures) < total_steps:
                    futures.add(executor.submit(step_task, cur_node))
    finally:
        # Cancel pending futures that have not started.
        for fut in list(futures):
            fut.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

    # Finalization phase (save artifacts and exit gracefully).
    try:
        save_run(cfg, journal)
        if timed_out:
            logger.info(journal_to_string_tree(journal))
            emit_event("pipeline", "RUN_FINALIZED_AFTER_TIMEOUT", completed=int(completed), total_steps=int(total_steps))
    except Exception as e:
        logger.warning(f"Final save_run failed: {e}")
        
    interpreter.cleanup_session(-1)
    emit_event("pipeline", "CLEANUP_COMPLETED")


if __name__ == "__main__":    
    run()
