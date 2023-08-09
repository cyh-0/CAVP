import torch.distributed as dist
import torch.utils.data.distributed
import torch
from loguru import logger
import builtins


def supress_printer(ddp, local_rank):
    if ddp and local_rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
        logger._log = print_pass

def barrier(ddp):
    if ddp:
        dist.barrier()
    return
