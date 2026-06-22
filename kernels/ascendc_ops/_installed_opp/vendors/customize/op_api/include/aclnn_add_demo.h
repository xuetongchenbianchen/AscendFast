
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ADD_DEMO_H_
#define ACLNN_ADD_DEMO_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnAddDemoGetWorkspaceSize
 * parameters :
 * x : required
 * y : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAddDemoGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *y,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnAddDemo
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAddDemo(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
