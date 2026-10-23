"""Position / velocity Kalman filter for the post-failure spin (IDEA-022 step B).

EKF2's velocity goes wrong in the spin because its tilt does. With a good attitude from the gyro, velocity and
position are a linear problem: predict with the accelerometer rotated to the world frame, correct with GPS position
and velocity. State [p, v] NED, independent axes. No ROS, so the offline check (bench/estimation.py kf) runs the same code.
"""
import numpy as np

G_NED = np.array([0.0, 0.0, 9.81])
ACC_SIGMA = 1.0          # [m/s^2] accelerometer + attitude error, as process noise
GPS_POS_SIGMA = 0.5      # [m]
GPS_VEL_SIGMA = 0.2      # [m/s]


class PosVelKF:
	def __init__(self, p, v):
		self.x = np.array([p, v], float)            # (2, 3): row 0 position, row 1 velocity, per axis
		self.P = np.tile(np.diag([0.1, 0.05]), (3, 1, 1))

	def predict(self, r_ned_frd, f_frd, dt):
		"""f_frd: accelerometer specific force [m/s^2]; acceleration = R f + g."""
		a = r_ned_frd @ f_frd + G_NED
		self.x[0] += self.x[1] * dt + 0.5 * a * dt * dt
		self.x[1] += a * dt
		F = np.array([[1.0, dt], [0.0, 1.0]])
		Q = ACC_SIGMA ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt * dt]])
		self.P = F @ self.P @ F.T + Q

	def update_gps(self, p, v):
		for k, (z, sigma) in enumerate(((p, GPS_POS_SIGMA), (v, GPS_VEL_SIGMA))):
			s = self.P[:, k, k] + sigma ** 2                  # (3,) innovation variance per axis
			gain = self.P[:, :, k] / s[:, None]               # (3, 2)
			innov = np.asarray(z, float) - self.x[k]
			self.x += (gain * innov[:, None]).T
			self.P -= gain[:, :, None] * self.P[:, k, None, :]

	@property
	def p(self):
		return self.x[0].copy()

	@property
	def v(self):
		return self.x[1].copy()


if __name__ == '__main__':
	# self-check: constant acceleration with noisy GPS, filter must track velocity to < 0.1 m/s
	rng = np.random.default_rng(0)
	kf, a_true, dt = PosVelKF(np.zeros(3), np.zeros(3)), np.array([0.5, -0.3, 0.2]), 0.004
	for k in range(1, 2501):
		t = k * dt
		kf.predict(np.eye(3), a_true - G_NED + rng.normal(0, 0.3, 3), dt)
		if k % 8 == 0:
			kf.update_gps(0.5 * a_true * t * t + rng.normal(0, 0.5, 3), a_true * t + rng.normal(0, 0.2, 3))
	err = np.abs(kf.v - a_true * t).max()
	print(f'velocity error after 10 s: {err:.3f} m/s')
	assert err < 0.1
