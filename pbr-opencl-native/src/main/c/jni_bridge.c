#include <jni.h>
#include "ocl_backend.h"
#include "vit_ocl_backend.h"
#include <stdlib.h>

JNIEXPORT jlong JNICALL Java_com_mc_pbr_opencl_CLNative_create(
    JNIEnv* env, jclass clazz, jintArray jLayerSizes, jlong seed) {
    jsize len = (*env)->GetArrayLength(env, jLayerSizes);
    int* layerSizes = (int*)malloc(len * sizeof(int));
    (*env)->GetIntArrayRegion(env, jLayerSizes, 0, len, layerSizes);
    OCLBackend* backend = ocl_backend_create(layerSizes, len - 1, (long)seed);
    free(layerSizes);
    return (jlong)(intptr_t)backend;
}

JNIEXPORT jlong JNICALL Java_com_mc_pbr_opencl_CLNative_createWithWeights(
    JNIEnv* env, jclass clazz, jintArray jLayerSizes,
    jfloatArray jWeights, jfloatArray jBiases) {
    jsize len = (*env)->GetArrayLength(env, jLayerSizes);
    int* layerSizes = (int*)malloc(len * sizeof(int));
    (*env)->GetIntArrayRegion(env, jLayerSizes, 0, len, layerSizes);
    float* weights = NULL;
    float* biases = NULL;
    if (jWeights) weights = (*env)->GetFloatArrayElements(env, jWeights, NULL);
    if (jBiases) biases = (*env)->GetFloatArrayElements(env, jBiases, NULL);
    OCLBackend* backend = ocl_backend_create_with_weights(layerSizes, len - 1, weights, biases);
    if (weights) (*env)->ReleaseFloatArrayElements(env, jWeights, weights, JNI_ABORT);
    if (biases) (*env)->ReleaseFloatArrayElements(env, jBiases, biases, JNI_ABORT);
    free(layerSizes);
    return (jlong)(intptr_t)backend;
}

JNIEXPORT void JNICALL Java_com_mc_pbr_opencl_CLNative_destroy(
    JNIEnv* env, jclass clazz, jlong handle) {
    OCLBackend* backend = (OCLBackend*)(intptr_t)handle;
    if (backend) ocl_backend_destroy(backend);
}

JNIEXPORT void JNICALL Java_com_mc_pbr_opencl_CLNative_forward(
    JNIEnv* env, jclass clazz, jlong handle,
    jfloatArray jInput, jfloatArray jOutput, jint batchSize) {
    OCLBackend* backend = (OCLBackend*)(intptr_t)handle;
    if (!backend) return;
    float* input = (*env)->GetFloatArrayElements(env, jInput, NULL);
    float* output = (*env)->GetFloatArrayElements(env, jOutput, NULL);
    ocl_backend_forward(backend, input, output, batchSize);
    (*env)->ReleaseFloatArrayElements(env, jInput, input, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, jOutput, output, 0);
}

JNIEXPORT void JNICALL Java_com_mc_pbr_opencl_CLNative_backward(
    JNIEnv* env, jclass clazz, jlong handle,
    jfloatArray jInput, jfloatArray jLabel, jfloatArray jGradOutput, jint batchSize) {
    OCLBackend* backend = (OCLBackend*)(intptr_t)handle;
    if (!backend) return;
    float* input = (*env)->GetFloatArrayElements(env, jInput, NULL);
    float* label = (*env)->GetFloatArrayElements(env, jLabel, NULL);
    float* gradOutput = (*env)->GetFloatArrayElements(env, jGradOutput, NULL);
    ocl_backend_backward(backend, input, label, gradOutput, batchSize);
    (*env)->ReleaseFloatArrayElements(env, jInput, input, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, jLabel, label, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, jGradOutput, gradOutput, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_com_mc_pbr_opencl_CLNative_update(
    JNIEnv* env, jclass clazz, jlong handle,
    jfloatArray jGradWeights, jfloatArray jGradBiases,
    jint batchSize, jfloat lr, jfloat momentum) {
    OCLBackend* backend = (OCLBackend*)(intptr_t)handle;
    if (!backend) return;
    float* gradWeights = NULL;
    float* gradBiases = NULL;
    if (jGradWeights) gradWeights = (*env)->GetFloatArrayElements(env, jGradWeights, NULL);
    if (jGradBiases) gradBiases = (*env)->GetFloatArrayElements(env, jGradBiases, NULL);
    ocl_backend_update(backend, gradWeights, gradBiases, batchSize, lr, momentum);
    if (gradWeights) (*env)->ReleaseFloatArrayElements(env, jGradWeights, gradWeights, JNI_ABORT);
    if (gradBiases) (*env)->ReleaseFloatArrayElements(env, jGradBiases, gradBiases, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_com_mc_pbr_opencl_CLNative_zeroGradients(
    JNIEnv* env, jclass clazz, jlong handle) {
    OCLBackend* backend = (OCLBackend*)(intptr_t)handle;
    if (backend) ocl_backend_zero_gradients(backend);
}

JNIEXPORT jfloatArray JNICALL Java_com_mc_pbr_opencl_CLNative_getWeights(
    JNIEnv* env, jclass clazz, jlong handle) {
    OCLBackend* backend = (OCLBackend*)(intptr_t)handle;
    if (!backend) return NULL;
    int total = ocl_backend_get_total_weights(backend);
    jfloatArray result = (*env)->NewFloatArray(env, total);
    float* arr = (*env)->GetFloatArrayElements(env, result, NULL);
    ocl_backend_get_weights(backend, arr);
    (*env)->ReleaseFloatArrayElements(env, result, arr, 0);
    return result;
}

JNIEXPORT jfloatArray JNICALL Java_com_mc_pbr_opencl_CLNative_getBiases(
    JNIEnv* env, jclass clazz, jlong handle) {
    OCLBackend* backend = (OCLBackend*)(intptr_t)handle;
    if (!backend) return NULL;
    int total = ocl_backend_get_total_biases(backend);
    jfloatArray result = (*env)->NewFloatArray(env, total);
    float* arr = (*env)->GetFloatArrayElements(env, result, NULL);
    ocl_backend_get_biases(backend, arr);
    (*env)->ReleaseFloatArrayElements(env, result, arr, 0);
    return result;
}

JNIEXPORT void JNICALL Java_com_mc_pbr_opencl_CLNative_setWeights(
    JNIEnv* env, jclass clazz, jlong handle, jfloatArray jWeights) {
    OCLBackend* backend = (OCLBackend*)(intptr_t)handle;
    if (!backend) return;
    float* weights = (*env)->GetFloatArrayElements(env, jWeights, NULL);
    ocl_backend_set_weights(backend, weights);
    (*env)->ReleaseFloatArrayElements(env, jWeights, weights, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_com_mc_pbr_opencl_CLNative_setBiases(
    JNIEnv* env, jclass clazz, jlong handle, jfloatArray jBiases) {
    OCLBackend* backend = (OCLBackend*)(intptr_t)handle;
    if (!backend) return;
    float* biases = (*env)->GetFloatArrayElements(env, jBiases, NULL);
    ocl_backend_set_biases(backend, biases);
    (*env)->ReleaseFloatArrayElements(env, jBiases, biases, JNI_ABORT);
}

JNIEXPORT jlong JNICALL Java_com_mc_pbr_opencl_CLNative_createViT(
    JNIEnv* env, jclass clazz, jint embedDim, jint numLayers, jint numHeads, jint mlpDim, jint seqLen, jint inChannels, jlong seed) {
    VitBackend* backend = vit_backend_create(embedDim, numLayers, numHeads, mlpDim, seqLen, inChannels, (long)seed);
    return (jlong)(intptr_t)backend;
}

JNIEXPORT jlong JNICALL Java_com_mc_pbr_opencl_CLNative_createViTWithWeights(
    JNIEnv* env, jclass clazz, jint embedDim, jint numLayers, jint numHeads, jint mlpDim, jint seqLen, jint inChannels,
    jfloatArray jWeights, jfloatArray jBiases) {
    float* weights = NULL;
    float* biases = NULL;
    if (jWeights) weights = (*env)->GetFloatArrayElements(env, jWeights, NULL);
    if (jBiases) biases = (*env)->GetFloatArrayElements(env, jBiases, NULL);
    VitBackend* backend = vit_backend_create_with_weights(embedDim, numLayers, numHeads, mlpDim, seqLen, inChannels, weights, biases);
    if (weights) (*env)->ReleaseFloatArrayElements(env, jWeights, weights, JNI_ABORT);
    if (biases) (*env)->ReleaseFloatArrayElements(env, jBiases, biases, JNI_ABORT);
    return (jlong)(intptr_t)backend;
}

JNIEXPORT void JNICALL Java_com_mc_pbr_opencl_CLNative_destroyViT(
    JNIEnv* env, jclass clazz, jlong handle) {
    VitBackend* backend = (VitBackend*)(intptr_t)handle;
    if (backend) vit_backend_destroy(backend);
}

JNIEXPORT void JNICALL Java_com_mc_pbr_opencl_CLNative_forwardViT(
    JNIEnv* env, jclass clazz, jlong handle, jfloatArray jInput, jfloatArray jOutput, jint batchSize) {
    VitBackend* backend = (VitBackend*)(intptr_t)handle;
    if (!backend) return;
    float* input = (*env)->GetFloatArrayElements(env, jInput, NULL);
    float* output = (*env)->GetFloatArrayElements(env, jOutput, NULL);
    vit_backend_forward(backend, input, output, batchSize);
    (*env)->ReleaseFloatArrayElements(env, jInput, input, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, jOutput, output, 0);
}

JNIEXPORT void JNICALL Java_com_mc_pbr_opencl_CLNative_backwardViT(
    JNIEnv* env, jclass clazz, jlong handle, jfloatArray jInput, jfloatArray jLabel, jfloatArray jGradOutput, jint batchSize) {
    VitBackend* backend = (VitBackend*)(intptr_t)handle;
    if (!backend) return;
    float* input = (*env)->GetFloatArrayElements(env, jInput, NULL);
    float* label = (*env)->GetFloatArrayElements(env, jLabel, NULL);
    float* gradOutput = (*env)->GetFloatArrayElements(env, jGradOutput, NULL);
    vit_backend_backward(backend, input, label, gradOutput, batchSize);
    (*env)->ReleaseFloatArrayElements(env, jInput, input, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, jLabel, label, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, jGradOutput, gradOutput, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_com_mc_pbr_opencl_CLNative_zeroGradientsViT(
    JNIEnv* env, jclass clazz, jlong handle) {
    VitBackend* backend = (VitBackend*)(intptr_t)handle;
    if (backend) vit_backend_zero_gradients(backend);
}

JNIEXPORT jfloatArray JNICALL Java_com_mc_pbr_opencl_CLNative_getViTWeights(
    JNIEnv* env, jclass clazz, jlong handle) {
    VitBackend* backend = (VitBackend*)(intptr_t)handle;
    if (!backend) return NULL;
    int total = vit_backend_get_total_weights(backend);
    jfloatArray result = (*env)->NewFloatArray(env, total);
    float* arr = (*env)->GetFloatArrayElements(env, result, NULL);
    vit_backend_get_weights(backend, arr);
    (*env)->ReleaseFloatArrayElements(env, result, arr, 0);
    return result;
}

JNIEXPORT jfloatArray JNICALL Java_com_mc_pbr_opencl_CLNative_getViTBiases(
    JNIEnv* env, jclass clazz, jlong handle) {
    VitBackend* backend = (VitBackend*)(intptr_t)handle;
    if (!backend) return NULL;
    int total = vit_backend_get_total_biases(backend);
    jfloatArray result = (*env)->NewFloatArray(env, total);
    float* arr = (*env)->GetFloatArrayElements(env, result, NULL);
    vit_backend_get_biases(backend, arr);
    (*env)->ReleaseFloatArrayElements(env, result, arr, 0);
    return result;
}

JNIEXPORT void JNICALL Java_com_mc_pbr_opencl_CLNative_setViTWeights(
    JNIEnv* env, jclass clazz, jlong handle, jfloatArray jWeights) {
    VitBackend* backend = (VitBackend*)(intptr_t)handle;
    if (!backend) return;
    float* weights = (*env)->GetFloatArrayElements(env, jWeights, NULL);
    vit_backend_set_weights(backend, weights);
    (*env)->ReleaseFloatArrayElements(env, jWeights, weights, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_com_mc_pbr_opencl_CLNative_setViTBiases(
    JNIEnv* env, jclass clazz, jlong handle, jfloatArray jBiases) {
    VitBackend* backend = (VitBackend*)(intptr_t)handle;
    if (!backend) return;
    float* biases = (*env)->GetFloatArrayElements(env, jBiases, NULL);
    vit_backend_set_biases(backend, biases);
    (*env)->ReleaseFloatArrayElements(env, jBiases, biases, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_com_mc_pbr_opencl_CLNative_adamwUpdateViT(
        JNIEnv* env, jclass clazz, jlong handle, jint batchSize, jfloat lr, jfloat beta1, jfloat beta2, jfloat epsilon, jfloat weightDecay, jint step) {
    VitBackend* backend = (VitBackend*)(intptr_t)handle;
    if (!backend) return;
    vit_backend_adamw_update(backend, batchSize, lr, beta1, beta2, epsilon, weightDecay, step);
}

JNIEXPORT void JNICALL Java_com_mc_pbr_opencl_CLNative_clipGradientsViT(
        JNIEnv* env, jclass clazz, jlong handle, jfloat maxNorm) {
    VitBackend* backend = (VitBackend*)(intptr_t)handle;
    if (!backend) return;
    vit_backend_clip_gradients(backend, maxNorm);
}

JNIEXPORT jfloat JNICALL Java_com_mc_pbr_opencl_CLNative_mppForwardViT(
        JNIEnv* env, jclass clazz, jlong handle, jintArray jMaskIndices, jintArray jTargets, jint batchSize, jint numMasked, jint numClasses) {
    VitBackend* backend = (VitBackend*)(intptr_t)handle;
    if (!backend) return 0.0f;
    int* maskIndices = (*env)->GetIntArrayElements(env, jMaskIndices, NULL);
    int* targets = (*env)->GetIntArrayElements(env, jTargets, NULL);
    float loss = vit_backend_mpp_forward(backend, maskIndices, targets, batchSize, numMasked, numClasses);
    (*env)->ReleaseIntArrayElements(env, jMaskIndices, maskIndices, JNI_ABORT);
    (*env)->ReleaseIntArrayElements(env, jTargets, targets, JNI_ABORT);
    return loss;
}

JNIEXPORT void JNICALL Java_com_mc_pbr_opencl_CLNative_mppBackwardViT(
        JNIEnv* env, jclass clazz, jlong handle, jintArray jMaskIndices, jintArray jTargets, jint batchSize, jint numMasked, jint numClasses, jfloat lossScale) {
    VitBackend* backend = (VitBackend*)(intptr_t)handle;
    if (!backend) return;
    int* maskIndices = (*env)->GetIntArrayElements(env, jMaskIndices, NULL);
    int* targets = (*env)->GetIntArrayElements(env, jTargets, NULL);
    vit_backend_mpp_backward(backend, maskIndices, targets, batchSize, numMasked, numClasses, lossScale);
    (*env)->ReleaseIntArrayElements(env, jMaskIndices, maskIndices, JNI_ABORT);
    (*env)->ReleaseIntArrayElements(env, jTargets, targets, JNI_ABORT);
}

JNIEXPORT jfloatArray JNICALL Java_com_mc_pbr_opencl_CLNative_getViTGradients(
    JNIEnv* env, jclass clazz, jlong handle) {
    VitBackend* backend = (VitBackend*)(intptr_t)handle;
    if (!backend) return NULL;
    int total = vit_backend_get_total_weights(backend);
    jfloatArray result = (*env)->NewFloatArray(env, total);
    float* arr = (*env)->GetFloatArrayElements(env, result, NULL);
    vit_backend_get_gradients(backend, arr);
    (*env)->ReleaseFloatArrayElements(env, result, arr, 0);
    return result;
}