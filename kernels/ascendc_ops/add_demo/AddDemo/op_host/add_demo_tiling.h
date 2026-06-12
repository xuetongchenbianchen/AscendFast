
#include "register/tilingdata_base.h"

namespace optiling {
// tiling 结构：host 侧 TilingFunc 算好这些值，传给 device 侧 kernel。
// totalLength = 元素总数；tileNum = 每个核内再切成几块（流水并行用）。
BEGIN_TILING_DATA_DEF(AddDemoTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddDemo, AddDemoTilingData)
}
