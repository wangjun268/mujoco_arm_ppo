#!/usr/bin/env python3
"""旧入口：现在只是 ``flows.full_task`` 的薄包装。

真正的搬运流程按阶段拆在 ``flows/`` 下，一个文件一段：

    flows/prepare.py   回到预抓取位
    flows/grasp.py     闭手抓住
    flows/lift.py      抬起
    flows/carry.py     向前搬运
    flows/place.py     放回台面
    flows/release.py   松手
    flows/full_task.py 串起来（本文件就是它）

    python3 demo_plan.py                 # = python3 -m take_box_flows.full_task
    python3 demo_plan.py --video         # 同时存 runs/take_box/full_task.gif
    python3 demo_plan.py --strict        # 约束越界直接抛异常
    python3 demo_plan.py --forward 0.10 --side 0.06 --yaw 0.10
"""

from __future__ import annotations

from take_box_flows.full_task import main

if __name__ == "__main__":
    raise SystemExit(main())
