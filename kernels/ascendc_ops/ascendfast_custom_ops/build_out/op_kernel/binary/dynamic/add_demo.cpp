#include "kernel_operator.h"

// device 侧 kernel：z = x + y，逐元素。Ascend C 的标准三段流水范式：
// CopyIn(GM→Local) → Compute(矢量指令) → CopyOut(Local→GM)。
// 写自己的算子时，主要改 Compute 里的 Ascend C 指令；搬运/切分骨架基本不变。
constexpr int32_t BUFFER_NUM = 1;

template <class T>
class KernelAddDemo {
public:
  __aicore__ inline KernelAddDemo() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                              uint32_t totalLength, uint32_t tileNum) {
    // 每个核负责 totalLength / 核数 个元素。
    this->blockLength = totalLength / AscendC::GetBlockNum();
    this->tileNum = tileNum;
    this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

    // 本核的 GM 起始地址 = 全局基址 + 本核偏移。
    xGm.SetGlobalBuffer((__gm__ T*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
    yGm.SetGlobalBuffer((__gm__ T*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
    zGm.SetGlobalBuffer((__gm__ T*)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(T));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(T));
  }

  __aicore__ inline void Process() {
    int32_t loopCount = this->tileNum * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
      CopyIn(i);
      Compute(i);
      CopyOut(i);
    }
  }

private:
  __aicore__ inline void CopyIn(int32_t progress) {
    AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
    AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
  }

  __aicore__ inline void Compute(int32_t progress) {
    AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inQueueY.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
    // ★ 算法核心:换成别的算子时改这一行(及前后所需的中间 buffer)。
    // Add 要求 count 参数是 const int32_t&,所以显式转换并取引用。
    const int32_t count = static_cast<int32_t>(this->tileLength);
    AscendC::Add(zLocal, xLocal, yLocal, count);
    outQueueZ.EnQue<T>(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
  }

  __aicore__ inline void CopyOut(int32_t progress) {
    AscendC::LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
    AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
    outQueueZ.FreeTensor(zLocal);
  }

private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
  AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
  AscendC::GlobalTensor<T> xGm, yGm, zGm;
  uint32_t blockLength;
  uint32_t tileNum;
  uint32_t tileLength;
};

extern "C" __global__ __aicore__ void add_demo(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                               GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  // DTYPE_X 由框架按 IR 的 dtype 注入（fp16/fp32 各编一份）。
  KernelAddDemo<DTYPE_X> op;
  op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
  op.Process();
}
