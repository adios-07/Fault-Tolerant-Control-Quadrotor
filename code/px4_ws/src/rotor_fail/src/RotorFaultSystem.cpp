// Gazebo system that injects rotor faults below PX4, so no flight-mode change is needed.
// Replaces PX4's MotorFailurePlugin (which only supports a full stop).
//
// Command topic /model/<model>/motor_fault (gz.msgs.StringMsg): "<motor> <mode> <intensity>", e.g. "3 prop 1.0"
//   motor      1-4 (rotor_0..3), 0 clears the fault
//   mode       stop : joint speed forced to 0 (motor/ESC dead; RPM drops)
//              weak : speed scaled so thrust = (1 - intensity) * normal (weak motor; RPM drops)
//              prop : intensity * thrust and yaw drag cancelled, rotor keeps spinning
//                     (lost/damaged propeller; RPM stays normal, ESC cannot see it)
//   intensity  fraction of thrust lost, 0..1 (ignored for stop)
//
// Must be declared AFTER the MulticopterMotorModel plugins: it edits what they wrote this step.

#include <algorithm>
#include <cmath>
#include <mutex>
#include <sstream>
#include <string>

#include <gz/msgs/stringmsg.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/components/JointVelocity.hh>
#include <gz/sim/components/JointVelocityCmd.hh>
#include <gz/transport/Node.hh>

namespace rotor_fail
{
using namespace gz;
using namespace sim;

// From models/custom_x500/model.sdf
constexpr double KF = 1.18e-05;      // motorConstant: thrust = KF * speed^2
constexpr double KM = 0.016;         // momentConstant: yaw drag = KM * thrust
constexpr double SLOWDOWN = 10.0;    // rotorVelocitySlowdownSim: joint speed * 10 = rotor speed

class RotorFaultSystem : public System, public ISystemConfigure, public ISystemPreUpdate
{
public:
	void Configure(const Entity &_entity, const std::shared_ptr<const sdf::Element> &,
		       EntityComponentManager &_ecm, EventManager &) override
	{
		model_ = Model(_entity);
		const std::string topic = "/model/" + model_.Name(_ecm) + "/motor_fault";
		node_.Subscribe(topic, &RotorFaultSystem::OnCommand, this);
		gzmsg << "[RotorFault] listening on " << topic << std::endl;
	}

	void PreUpdate(const UpdateInfo &_info, EntityComponentManager &_ecm) override
	{
		if (_info.paused) {
			return;
		}
		if (base_ == kNullEntity) {
			base_ = model_.LinkByName(_ecm, "base_link");
			for (int i = 0; i < 4; i++) {
				joints_[i] = model_.JointByName(_ecm, "rotor_" + std::to_string(i) + "_joint");
				links_[i] = model_.LinkByName(_ecm, "rotor_" + std::to_string(i));
			}
		}

		int motor; std::string mode; double k;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			motor = motor_; mode = mode_; k = intensity_;
		}
		if (motor < 1 || motor > 4) {
			return;
		}
		const int i = motor - 1;

		if (mode == "stop" || mode == "weak") {
			auto cmd = _ecm.Component<components::JointVelocityCmd>(joints_[i]);
			if (cmd && !cmd->Data().empty()) {
				// thrust ~ speed^2, so scale speed by sqrt to scale thrust linearly
				cmd->Data()[0] *= (mode == "stop") ? 0.0 : std::sqrt(1.0 - k);
			}
			return;
		}

		// prop: undo the motor model's force and yaw drag for this step (same formula it used)
		auto jv = _ecm.Component<components::JointVelocity>(joints_[i]);
		if (!jv || jv->Data().empty()) {
			return;
		}
		const double rv = jv->Data()[0] * SLOWDOWN;
		const double dir = (rv > 0) - (rv < 0);  // = turning direction while spinning normally
		const double thrust = KF * rv * rv * k;
		Link rotor(links_[i]);
		const auto pose = rotor.WorldPose(_ecm);
		if (!pose) {
			return;
		}
		rotor.AddWorldForce(_ecm, pose->Rot().RotateVector(math::Vector3d(0, 0, -thrust)));
		Link(base_).AddWorldWrench(_ecm, math::Vector3d::Zero,
					   pose->Rot().RotateVector(math::Vector3d(0, 0, dir * thrust * KM)));
	}

private:
	void OnCommand(const msgs::StringMsg &_msg)
	{
		std::istringstream in(_msg.data());
		int motor = 0; std::string mode = "stop"; double k = 1.0;
		in >> motor >> mode >> k;
		if (mode != "stop" && mode != "weak" && mode != "prop") {
			gzerr << "[RotorFault] unknown mode '" << mode << "' (stop|weak|prop)" << std::endl;
			return;
		}
		k = std::min(std::max(k, 0.0), 1.0);
		std::lock_guard<std::mutex> lock(mutex_);
		motor_ = motor; mode_ = mode; intensity_ = k;
		if (motor >= 1 && motor <= 4) {
			gzwarn << "[RotorFault] motor " << motor << " " << mode << " intensity " << k << std::endl;
		} else {
			gzwarn << "[RotorFault] fault cleared" << std::endl;
		}
	}

	Model model_{kNullEntity};
	Entity base_{kNullEntity};
	Entity joints_[4]{kNullEntity, kNullEntity, kNullEntity, kNullEntity};
	Entity links_[4]{kNullEntity, kNullEntity, kNullEntity, kNullEntity};
	transport::Node node_;
	std::mutex mutex_;
	int motor_{0};
	std::string mode_{"stop"};
	double intensity_{1.0};
};
}  // namespace rotor_fail

GZ_ADD_PLUGIN(rotor_fail::RotorFaultSystem, gz::sim::System,
	      gz::sim::ISystemConfigure, gz::sim::ISystemPreUpdate)
