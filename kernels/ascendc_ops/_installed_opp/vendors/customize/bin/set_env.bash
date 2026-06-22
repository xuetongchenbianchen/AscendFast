#!/bin/bash
export ASCEND_CUSTOM_OPP_PATH=/models/share/userdata/cb/AscendFast/kernels/ascendc_ops/_installed_opp/vendors/customize:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/models/share/userdata/cb/AscendFast/kernels/ascendc_ops/_installed_opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
