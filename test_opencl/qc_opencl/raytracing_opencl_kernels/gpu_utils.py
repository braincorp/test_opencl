# ============================================================================
# Copyright 2016 BRAIN Corporation. All rights reserved. This software is
# provided to you under BRAIN Corporation's Beta License Agreement and
# your use of the software is governed by the terms of that Beta License
# Agreement, found at http://www.braincorporation.com/betalicense.
# ============================================================================

import os


def get_opencl_kernel_path():
    return os.path.dirname(os.path.abspath(__file__))
