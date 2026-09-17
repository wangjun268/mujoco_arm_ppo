from glob import glob
import os

from setuptools import find_packages, setup

package_name = "pro7_pick_place_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        # All of config/ (parameters, the rviz layout, the robot-stack list).
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="wj",
    maintainer_email="wj@example.com",
    description="ROS 2 node around the MuJoCo Pro7 pick & place cell",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "pick_place_node = pro7_pick_place_ros.entry:run_node",
            "pick_place_client = pro7_pick_place_ros.entry:run_client",
        ],
    },
)
