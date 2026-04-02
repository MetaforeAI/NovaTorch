#pragma once
#include "./nova_compute.h"
#include <cstdint>

/**
 * NovaCommandBatch - Record multiple compute dispatches into a single
 * command buffer submission.
 *
 * Instead of submit+wait per dispatch (~20us overhead each),
 * record N dispatches with barriers, submit once, wait once.
 * For N=2000, saves ~40ms of pure overhead per training step.
 *
 * Usage:
 *   NovaCommandBatch batch(compute);
 *   batch.begin();
 *   batch.dispatch(pipeline1, layout1, desc_set1, 64);
 *   batch.barrier();
 *   batch.dispatch(pipeline2, layout2, desc_set2, 64);
 *   batch.submit();  // one submit + one fence wait for all dispatches
 */
class NovaCommandBatch {
public:
    explicit NovaCommandBatch(NovaCompute& compute);
    ~NovaCommandBatch();

    // Non-copyable, non-movable
    NovaCommandBatch(const NovaCommandBatch&) = delete;
    NovaCommandBatch& operator=(const NovaCommandBatch&) = delete;

    // Begin recording into the calling thread's command buffer.
    void begin();

    // Record a compute dispatch (pipeline bind + descriptor bind + dispatch)
    void dispatch(VkPipeline pipeline, VkPipelineLayout layout,
                  VkDescriptorSet desc_set,
                  uint32_t groups_x, uint32_t groups_y = 1,
                  uint32_t groups_z = 1);

    // Record a dispatch with push constants
    void dispatch(VkPipeline pipeline, VkPipelineLayout layout,
                  VkDescriptorSet desc_set,
                  const void* push_data, uint32_t push_size,
                  uint32_t groups_x, uint32_t groups_y = 1,
                  uint32_t groups_z = 1);

    // Insert a full compute-to-compute memory barrier.
    // Ensures all shader writes from previous dispatches are visible
    // to shader reads in subsequent dispatches.
    void barrier();

    // Record a buffer-to-buffer copy
    void copy(VkBuffer src, VkBuffer dst,
              VkDeviceSize src_offset, VkDeviceSize dst_offset,
              VkDeviceSize size);

    // Record a buffer fill
    void fill(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
              uint32_t value = 0);

    // End recording, submit all commands, and wait for GPU completion.
    // Releases the compute mutex.
    void submit();

    // End recording and submit without waiting.
    // Must call wait() before the next begin() or destruction.
    void submitAsync();

    // Check if an async submission has completed (non-blocking)
    bool isComplete() const;

    // Block until async submission completes. Releases the compute mutex.
    void wait();

    // Query state
    bool isRecording() const { return recording_; }
    uint32_t dispatchCount() const { return dispatch_count_; }

private:
    NovaCompute& compute_;
    bool recording_ = false;
    uint32_t dispatch_count_ = 0;
};
