# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from stretch.agent.base import ManagedOperation


class GoHomeOperation(ManagedOperation):
    """Make the robot go home"""

    _successful = False

    def can_start(self) -> bool:
        return self.agent is not None

    def run(self) -> None:
        self.intro(f"Attempting to go home")
        self.agent.go_home()
        self._successful = True
        self.cheer(f"Done going home")

    def was_successful(self) -> bool:
        return self._successful
