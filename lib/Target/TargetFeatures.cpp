#include "triton/Target/TargetFeatures.h"

namespace mlir {
namespace triton {

TargetFeatureSet::TargetFeatureSet()
    : features_(static_cast<unsigned>(TargetFeature::NumFeatures), false),
      sharedMemoryCapacity_(0) {}

bool TargetFeatureSet::has(TargetFeature feature) const {
  unsigned idx = static_cast<unsigned>(feature);
  if (idx >= features_.size())
    return false;
  return features_[idx];
}

void TargetFeatureSet::add(TargetFeature feature) {
  unsigned idx = static_cast<unsigned>(feature);
  if (idx < features_.size())
    features_.set(idx);
}

void TargetFeatureSet::remove(TargetFeature feature) {
  unsigned idx = static_cast<unsigned>(feature);
  if (idx < features_.size())
    features_.reset(idx);
}

int TargetFeatureSet::getBestMMAVersion() const {
  // Check in descending order of preference
  if (has(TargetFeature::MMAv5))
    return 5;
  if (has(TargetFeature::MMAv3))
    return 3;
  if (has(TargetFeature::MMAv2))
    return 2;
  if (has(TargetFeature::MMAv1))
    return 1;
  return 0;
}

size_t TargetFeatureSet::getSharedMemoryCapacity() const {
  return sharedMemoryCapacity_;
}

void TargetFeatureSet::setSharedMemoryCapacity(size_t capacity) {
  sharedMemoryCapacity_ = capacity;
}

bool TargetFeatureSet::empty() const {
  return features_.none();
}

void TargetFeatureSet::clear() {
  features_.reset();
  sharedMemoryCapacity_ = 0;
}

} // namespace triton
} // namespace mlir
