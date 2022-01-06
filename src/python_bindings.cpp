#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

// FCL
#include <fcl/fcl.h>

// YAML
#include <yaml-cpp/yaml.h>

// local
#include "robots.h"
#include "robotStatePropagator.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ob = ompl::base;
namespace oc = ompl::control;

class RobotHelper
{
public:
  RobotHelper(const std::string& robotType)
  {
    ob::RealVectorBounds position_bounds(2);
    position_bounds.setLow(-2);
    position_bounds.setHigh(2);
    robot_ = create_robot(robotType, position_bounds);

    auto si = robot_->getSpaceInformation();
    si->getStateSpace()->setup();
    state_sampler_ = si->allocStateSampler();
    control_sampler_ = si->allocControlSampler();

    tmp_state_a_ = si->allocState();
    tmp_state_b_ = si->allocState();
    tmp_control_ = si->allocControl();
  }

  ~RobotHelper()
  {
    auto si = robot_->getSpaceInformation();
    si->freeState(tmp_state_a_);
    si->freeState(tmp_state_b_);
    si->freeControl(tmp_control_);
  }

  float distance(const std::vector<double> &stateA, const std::vector<double> &stateB)
  {
    auto si = robot_->getSpaceInformation();
    si->getStateSpace()->copyFromReals(tmp_state_a_, stateA);
    si->getStateSpace()->copyFromReals(tmp_state_b_, stateB);
    return si->distance(tmp_state_a_, tmp_state_b_);
  }

  std::vector<double> sampleStateUniform()
  {
    state_sampler_->sampleUniform(tmp_state_a_);
    auto si = robot_->getSpaceInformation();
    std::vector<double> reals;
    si->getStateSpace()->copyToReals(reals, tmp_state_a_);
    return reals;
  }

  std::vector<double> sampleControlUniform()
  {
    control_sampler_->sample(tmp_control_);
    auto si = robot_->getSpaceInformation();
    si->printControl(tmp_control_);
    const size_t dim = si->getControlSpace()->getDimension();
    std::vector<double> reals(dim);
    for (size_t d = 0; d < dim; ++d)
    {
      double *address = si->getControlSpace()->getValueAddressAtIndex(tmp_control_, d);
      reals[d] = *address;
    }
    return reals;
  }

  std::vector<double> step(
    const std::vector<double> &state,
    const std::vector<double>& action,
    double duration)
  {
    auto si = robot_->getSpaceInformation();
    si->getStateSpace()->copyFromReals(tmp_state_a_, state);

    const size_t dim = si->getControlSpace()->getDimension();
    assert(dim == action.size());
    for (size_t d = 0; d < dim; ++d) {
      double *address = si->getControlSpace()->getValueAddressAtIndex(tmp_control_, d);
      *address = action[d];
    }
    robot_->propagate(tmp_state_a_, tmp_control_, duration, tmp_state_b_);

    std::vector<double> reals;
    si->getStateSpace()->copyToReals(reals, tmp_state_b_);
    return reals;
  }

private: 
  std::shared_ptr<Robot> robot_;
  ob::StateSamplerPtr state_sampler_;
  oc::ControlSamplerPtr control_sampler_;
  ob::State* tmp_state_a_;
  ob::State* tmp_state_b_;
  oc::Control* tmp_control_;
};

class CollisionChecker
{
public:
  CollisionChecker()
    : tmp_state_(nullptr)
  {

  }

  ~CollisionChecker()
  {
    if (robot_ && tmp_state_) {
      auto si = robot_->getSpaceInformation();
      si->freeState(tmp_state_);
    }
  }

  void load(const std::string& filename)
  {
    if (robot_ && tmp_state_) {
      auto si = robot_->getSpaceInformation();
      si->freeState(tmp_state_);
    }

    YAML::Node env = YAML::LoadFile(filename);

    std::vector<fcl::CollisionObjectf *> obstacles;
    for (const auto &obs : env["environment"]["obstacles"])
    {
      if (obs["type"].as<std::string>() == "box")
      {
        const auto &size = obs["size"];
        std::shared_ptr<fcl::CollisionGeometryf> geom;
        geom.reset(new fcl::Boxf(size[0].as<float>(), size[1].as<float>(), 1.0));
        const auto &center = obs["center"];
        auto co = new fcl::CollisionObjectf(geom);
        co->setTranslation(fcl::Vector3f(center[0].as<float>(), center[1].as<float>(), 0));
        obstacles.push_back(co);
      }
      else
      {
        throw std::runtime_error("Unknown obstacle type!");
      }
    }
    env_.reset(new fcl::DynamicAABBTreeCollisionManagerf());
    // std::shared_ptr<fcl::BroadPhaseCollisionManagerf> bpcm_env(new fcl::NaiveCollisionManagerf());
    env_->registerObjects(obstacles);
    env_->setup();

    const auto &robot_node = env["robots"][0];
    auto robotType = robot_node["type"].as<std::string>();
    const auto &dims = env["environment"]["dimensions"];
    ob::RealVectorBounds position_bounds(2);
    position_bounds.setLow(0);
    position_bounds.setHigh(0, dims[0].as<double>());
    position_bounds.setHigh(1, dims[1].as<double>());
    robot_ = create_robot(robotType, position_bounds);

    auto si = robot_->getSpaceInformation();
    si->getStateSpace()->setup();

    tmp_state_ = si->allocState();
  }

  auto distance(const std::vector<double>& state)
  {

    auto si = robot_->getSpaceInformation();
    si->getStateSpace()->copyFromReals(tmp_state_, state);

    std::vector<fcl::DefaultDistanceData<float>> distance_data(robot_->numParts());
    size_t min_idx = 0;
    for (size_t part = 0; part < robot_->numParts(); ++part) {
      const auto &transform = robot_->getTransform(tmp_state_, part);
      fcl::CollisionObjectf robot(robot_->getCollisionGeometry(part)); //, robot_->getTransform(state));
      robot.setTranslation(transform.translation());
      robot.setRotation(transform.rotation());
      distance_data[part].request.enable_signed_distance = true;
      env_->distance(&robot, &distance_data[part], fcl::DefaultDistanceFunction<float>);
      if (distance_data[part].result.min_distance < distance_data[min_idx].result.min_distance) {
        min_idx = part;
      }
    }

    return std::make_tuple(
      distance_data[min_idx].result.min_distance,
      distance_data[min_idx].result.nearest_points[0],
      distance_data[min_idx].result.nearest_points[1]);
  }

private:
  std::shared_ptr<fcl::CollisionGeometryf> geom_;
  std::shared_ptr<fcl::BroadPhaseCollisionManagerf> env_;
  std::shared_ptr<Robot> robot_;
  ob::State *tmp_state_;
};

PYBIND11_MODULE(motionplanningutils, m)
{
  pybind11::class_<CollisionChecker>(m, "CollisionChecker")
      .def(pybind11::init())
      .def("load", &CollisionChecker::load)
      .def("distance", &CollisionChecker::distance);

  pybind11::class_<RobotHelper>(m, "RobotHelper")
      .def(pybind11::init<const std::string &>())
      .def("distance", &RobotHelper::distance)
      .def("sampleUniform", &RobotHelper::sampleStateUniform)
      .def("sampleControlUniform", &RobotHelper::sampleControlUniform)
      .def("step", &RobotHelper::step);
}
