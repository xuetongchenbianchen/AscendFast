#include <stdint.h>
#include <map>
#include <tuple>
#include <vector>
#include "graph/ascend_string.h"
#include "register/op_impl_registry.h"

extern gert::OpImplRegisterV2 op_impl_register_optiling_AddDemo;
extern uint8_t _binary_config_ascend910_93_add_demo_json_start;
extern uint8_t _binary_config_ascend910_93_add_demo_json_end;
extern uint8_t _binary_ascend910_93_add_demo_AddDemo_d58661ffc141e782e1c3a4565b4023b3_json_start;
extern uint8_t _binary_ascend910_93_add_demo_AddDemo_d58661ffc141e782e1c3a4565b4023b3_json_end;
extern uint8_t _binary_ascend910_93_add_demo_AddDemo_d58661ffc141e782e1c3a4565b4023b3_o_start;
extern uint8_t _binary_ascend910_93_add_demo_AddDemo_d58661ffc141e782e1c3a4565b4023b3_o_end;
extern uint8_t _binary_ascend910_93_add_demo_AddDemo_d6687bc497d24a97d769cd6b825c218e_json_start;
extern uint8_t _binary_ascend910_93_add_demo_AddDemo_d6687bc497d24a97d769cd6b825c218e_json_end;
extern uint8_t _binary_ascend910_93_add_demo_AddDemo_d6687bc497d24a97d769cd6b825c218e_o_start;
extern uint8_t _binary_ascend910_93_add_demo_AddDemo_d6687bc497d24a97d769cd6b825c218e_o_end;

#define AddDemo_OP_RESOURCES std::make_tuple<std::vector<void *>, \
    std::map<ge::AscendString, std::vector<std::tuple<const uint8_t *, const uint8_t *>>>, \
    std::vector<std::tuple<const uint8_t *, const uint8_t *>>>({nullptr, nullptr}, \
    { { "ascend910_93", {    { &_binary_config_ascend910_93_add_demo_json_start, \
      &_binary_config_ascend910_93_add_demo_json_end } , \
            { &_binary_ascend910_93_add_demo_AddDemo_d58661ffc141e782e1c3a4565b4023b3_json_start, \
      &_binary_ascend910_93_add_demo_AddDemo_d58661ffc141e782e1c3a4565b4023b3_json_end } , \
            { &_binary_ascend910_93_add_demo_AddDemo_d58661ffc141e782e1c3a4565b4023b3_o_start, \
      &_binary_ascend910_93_add_demo_AddDemo_d58661ffc141e782e1c3a4565b4023b3_o_end } , \
            { &_binary_ascend910_93_add_demo_AddDemo_d6687bc497d24a97d769cd6b825c218e_json_start, \
      &_binary_ascend910_93_add_demo_AddDemo_d6687bc497d24a97d769cd6b825c218e_json_end } , \
            { &_binary_ascend910_93_add_demo_AddDemo_d6687bc497d24a97d769cd6b825c218e_o_start, \
      &_binary_ascend910_93_add_demo_AddDemo_d6687bc497d24a97d769cd6b825c218e_o_end }  } } }, \
    {  })

#define AddDemo_RESOURCES {{"AddDemo", AddDemo_OP_RESOURCES}}