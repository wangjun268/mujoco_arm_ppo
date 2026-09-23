"""take_box 搬运流程：每个阶段一个文件，既能单独跑，也能串成完整任务。

    流程文件              动作
    prepare.py            回到预抓取位（IK 解出来的抓取位姿）
    grasp.py              闭手抓住（掌心贴住箱面即锁存）
    lift.py               抬起箱子
    carry.py              手臂向前搬运（可选横向/偏航）
    place.py              放回台面
    release.py            松手
    full_task.py          上面 6 步串起来的完整流程

每个文件都能单独运行（会自动先跑它前面的流程），例如：

    python3 -m take_box_flows.lift --height 0.12
    python3 -m take_box_flows.full_task --video
"""

from __future__ import annotations

from .carry import carry
from .grasp import grasp
from .lift import lift
from .place import place
from .prepare import prepare
from .release import release

__all__ = ["prepare", "grasp", "lift", "carry", "place", "release"]
