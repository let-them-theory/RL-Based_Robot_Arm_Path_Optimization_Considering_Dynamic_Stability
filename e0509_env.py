import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data as pd
import numpy as np
import os

URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "e0509", "e0509_2f85.urdf")

# ==============================================================================
# 두산로보틱스 E0509 (6축) + Robotiq 2F-85 (고정) 목표점 도달 강화학습 환경
# ==============================================================================
class E0509Env(gym.Env):
    JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    HOME_Q = np.array([0.0, 0.0, np.pi / 2, 0.0, np.pi / 2, 0.0])  # 두산 기본 홈 자세 (0, 0, 90, 0, 90, 0)

    def __init__(self, render=False, max_steps=500, stability=True, record=False):
        super(E0509Env, self).__init__()

        # False면 안정성 페널티(토크/베이스/스무딩/jerk)를 보상에서 제외 (비교 실험용). 비용 계산과 info 기록은 동일하게 수행
        self.stability = stability
        # True면 서브스텝(240Hz)마다 원시 역학 데이터를 self.trace에 기록 (평가 전용, 보상/동작에는 영향 없음)
        self.record = record
        self.trace = None

        self.render_mode = "human" if render else None
        if render:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(cameraDistance=1.6, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0.4, 0, 0.4])
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pd.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane = p.loadURDF("plane.urdf")
        # 로봇은 한 번만 로드하고 reset에서는 관절 상태만 초기화 (resetSimulation + loadURDF 반복 제거)
        self.robot = p.loadURDF(URDF_PATH, useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)

        name_to_idx = {p.getJointInfo(self.robot, j)[1].decode(): j for j in range(p.getNumJoints(self.robot))}
        link_to_idx = {p.getJointInfo(self.robot, j)[12].decode(): j for j in range(p.getNumJoints(self.robot))}
        self.joint_indices = [name_to_idx[n] for n in self.JOINT_NAMES]
        self.ee_link_index = link_to_idx["tcp"]  # 그리퍼 손가락 사이 (플랜지에서 156mm)
        self.n_joints = len(self.joint_indices)

        infos = [p.getJointInfo(self.robot, j) for j in self.joint_indices]
        self.ll = np.array([info[8] for info in infos])
        self.ul = np.array([info[9] for info in infos])
        self.joint_range = self.ul - self.ll
        self.max_forces = np.array([info[10] for info in infos])  # URDF 정격 토크 (194/194/194/66/66/66 Nm)

        # 베이스 반력 계산용: joint_1 이후 움직이는 모든 링크(팔+그리퍼)의 질량, 주축 관성, joint_1 원점(바닥에 고정된 점)
        # (질량 1e-4kg짜리 tcp/실리콘 패드 더미 링크는 제외)
        self.moving_links = [j for j in range(self.joint_indices[0], p.getNumJoints(self.robot))
                             if p.getDynamicsInfo(self.robot, j)[0] > 1e-3]
        dyn_infos = [p.getDynamicsInfo(self.robot, j) for j in self.moving_links]
        self.link_masses = np.array([d[0] for d in dyn_infos])
        self.link_inertias = np.array([d[2] for d in dyn_infos])
        self.base_origin = np.array(p.getLinkState(self.robot, self.joint_indices[0])[4])
        self.base_force_ref = self.link_masses.sum() * 9.81  # 정규화 기준: 팔+그리퍼 자중 (약 208N)
        self.base_moment_ref = self.max_forces[0]            # 정규화 기준: joint_1 정격 토크 (194Nm)

        # obs: q(6) + dq(6) + ee_pos(3) + rel_pos(3) + prev_action(6) + prev_prev_action(6)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4 * self.n_joints + 6,), dtype=np.float32)

        self.sim_step = 1./240.
        self.control_dt = 0.05
        self.n_substeps = int(self.control_dt / self.sim_step)
        self.max_steps = max_steps

        self.target_pos = None
        self.prev_action = np.zeros(self.n_joints)
        self.prev_prev_action = np.zeros(self.n_joints)

        self.target_marker = None
        if render:
            visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.005, rgbaColor=[0, 1, 0, 0.9])
            self.target_marker = p.createMultiBody(baseVisualShapeIndex=visual_id, basePosition=[0, 0, -1])

    def _ee_pos(self):
        # [0]은 링크 질량중심(COM), [4]가 URDF 링크 프레임 (IK 기준). 둘이 다른 링크가 있으므로 항상 [4] 사용
        return np.array(p.getLinkState(self.robot, self.ee_link_index)[4])

    def _get_obs(self):
        joint_states = p.getJointStates(self.robot, self.joint_indices)
        q = np.array([state[0] for state in joint_states])
        dq = np.array([state[1] for state in joint_states])

        ee_pos = self._ee_pos()
        rel_pos = self.target_pos - ee_pos
        return np.concatenate([q, dq, ee_pos, rel_pos, self.prev_action, self.prev_prev_action]).astype(np.float32)

    def _dynamic_torque_cost(self):
        """중력 보상분을 뺀 (운동에 의한) 토크를 정격 토크로 정규화한 제곱 평균"""
        joint_states = p.getJointStates(self.robot, self.joint_indices)
        q = [state[0] for state in joint_states]
        applied = np.array([state[3] for state in joint_states])
        gravity = np.array(p.calculateInverseDynamics(self.robot, q, [0.0] * self.n_joints, [0.0] * self.n_joints))
        return np.mean(np.square((applied - gravity) / self.max_forces))

    def _link_kinematics(self):
        """움직이는 링크들의 COM 위치(joint_1 원점 기준) r, 선속도 v, 각속도 omega, 자전 각운동량 R I R^T w"""
        states = p.getLinkStates(self.robot, self.moving_links, computeLinkVelocity=1)
        r = np.array([s[0] for s in states]) - self.base_origin  # COM 위치
        x, y, z, w = np.array([s[1] for s in states]).T           # COM(주축) 프레임 자세 쿼터니언
        v = np.array([s[6] for s in states])
        omega = np.array([s[7] for s in states])

        R = np.stack([
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
        ], axis=1)
        spin = np.einsum('nij,nj->ni', R, self.link_inertias * np.einsum('nji,nj->ni', R, omega))  # R I R^T w
        return r, v, omega, spin

    def _momentum(self):
        """joint_1 원점 기준 팔 전체의 선운동량 P, 각운동량 L (월드 좌표).
        베이스가 받는 하중 중 중력 외의 동적 성분 = (dP/dt, dL/dt)"""
        r, v, omega, spin = self._link_kinematics()
        P = self.link_masses @ v
        L = (self.link_masses[:, None] * np.cross(r, v) + spin).sum(axis=0)
        return P, L

    def _record_sample(self, P, L):
        """평가용 원시 데이터 1샘플 기록: 관절 상태/토크, 운동량, 팔 COM, 운동 에너지, TCP 위치/속도"""
        r, v, omega, spin = self._link_kinematics()
        joint_states = p.getJointStates(self.robot, self.joint_indices)
        q = [state[0] for state in joint_states]
        tcp = p.getLinkState(self.robot, self.ee_link_index, computeLinkVelocity=1)
        self.trace.append({
            "q": q,
            "dq": [state[1] for state in joint_states],
            "tau": [state[3] for state in joint_states],
            "tau_g": p.calculateInverseDynamics(self.robot, q, [0.0] * self.n_joints, [0.0] * self.n_joints),
            "P": P,
            "L": L,
            "com": self.base_origin + self.link_masses @ r / self.link_masses.sum(),
            "ke": 0.5 * self.link_masses @ np.sum(v * v, axis=1) + 0.5 * np.sum(omega * spin),
            "tcp_pos": tcp[4],
            "tcp_vel": tcp[6],
        })

    def _sample_valid_target(self):
        """IK 해가 관절 한계 안에 있고, 바닥과 충돌하지 않으며, 실제로 5mm 이내에 도달하는 목표만 채택"""
        for _ in range(50):
            target = np.array([
                self.np_random.uniform(0.2, 0.7),
                self.np_random.uniform(-0.5, 0.5),
                self.np_random.uniform(0.1, 0.8)
            ])

            for i, joint_idx in enumerate(self.joint_indices):
                p.resetJointState(self.robot, joint_idx, self.HOME_Q[i])

            # numpy 배열을 넘기면 PyBullet이 segfault → list로 변환
            joint_poses = np.array(p.calculateInverseKinematics(
                self.robot, self.ee_link_index, target.tolist(),
                lowerLimits=self.ll.tolist(), upperLimits=self.ul.tolist(),
                jointRanges=self.joint_range.tolist(), restPoses=self.HOME_Q.tolist(),
                maxNumIterations=30
            ))
            if np.any(joint_poses < self.ll) or np.any(joint_poses > self.ul):
                continue

            for i, joint_idx in enumerate(self.joint_indices):
                p.resetJointState(self.robot, joint_idx, joint_poses[i])

            actual_ee_pos = self._ee_pos()

            p.performCollisionDetection()
            if len(p.getContactPoints(bodyA=self.robot, bodyB=self.plane)) > 0:
                continue

            if np.linalg.norm(target - actual_ee_pos) < 0.005:
                return target

        return np.array([0.4, 0.0, 0.4])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.target_pos = self._sample_valid_target()

        if self.target_marker is not None:
            p.resetBasePositionAndOrientation(self.target_marker, self.target_pos, [0, 0, 0, 1])

        init_q = self.HOME_Q + self.np_random.uniform(-0.3, 0.3, size=self.n_joints)
        for i in range(self.n_joints):
            p.resetJointState(self.robot, self.joint_indices[i], init_q[i])

        self.step_cnt = 0
        self.prev_action = np.zeros(self.n_joints)
        self.prev_prev_action = np.zeros(self.n_joints)
        self.prev_momentum = self._momentum()
        if self.record:
            self.trace = []
            self._record_sample(*self.prev_momentum)
        return self._get_obs(), {}

    def step(self, action):
        self.step_cnt += 1
        action = np.asarray(action, dtype=np.float64)
        real_action = action * 0.03

        current_q = np.array([p.getJointState(self.robot, i)[0] for i in self.joint_indices])
        target_q = np.clip(current_q + real_action, self.ll, self.ul)

        p.setJointMotorControlArray(self.robot, self.joint_indices, p.POSITION_CONTROL,
                                    targetPositions=target_q.tolist(), forces=self.max_forces.tolist())

        # 토크/베이스 반력 비용은 마지막 서브스텝 하나가 아니라 모든 서브스텝에서 평균
        torque_cost = 0.0
        base_cost = 0.0
        for _ in range(self.n_substeps):
            p.stepSimulation()
            torque_cost += self._dynamic_torque_cost()

            P, L = self._momentum()
            base_force = (P - self.prev_momentum[0]) / self.sim_step
            base_moment = (L - self.prev_momentum[1]) / self.sim_step
            self.prev_momentum = (P, L)
            if self.record:
                self._record_sample(P, L)
            base_cost += (np.linalg.norm(base_force) / self.base_force_ref) ** 2 \
                       + (np.linalg.norm(base_moment) / self.base_moment_ref) ** 2
        torque_cost /= self.n_substeps
        base_cost /= self.n_substeps

        # 행동 = 관절 위치 증분(≈속도) → 1차 차분 ≈ 가속도, 2차 차분 ≈ jerk
        smooth_cost = np.mean(np.square(action - self.prev_action))
        jerk_cost = np.mean(np.square(action - 2 * self.prev_action + self.prev_prev_action))
        # obs의 prev/prev_prev_action은 이번 행동으로 갱신된 값 (다음 스텝 비용 계산과 일치)
        self.prev_prev_action = self.prev_action
        self.prev_action = action

        obs = self._get_obs()
        ee_pos = obs[2 * self.n_joints:2 * self.n_joints + 3]
        distance = np.linalg.norm(self.target_pos - ee_pos)

        # 스텝당 보상은 항상 ≤ 0 이어야 함. 양수면 목표 근처에 머물며 보상을 쌓는 것이
        # 도달(+150, 종료)보다 유리해져 5mm 바로 밖에서 멈추는 reward hacking이 발생함
        # = 원래 식 -20d + (0.02-d)*200 에서 상수 4를 뺀 것: 기울기 동일(바깥 20, 2cm 안 220), 2cm 경계에서 연속
        reward = -distance * 20.0 - 200.0 * min(distance, 0.02)

        # 안정성 항목 가중치 (미튜닝 시작값). 측정한 스텝당 평균 비용 [부드러운 IK 추종 / 최대속도 IK 추종]:
        #   토크 0.020 / 0.089, 베이스 0.49 / 2.34, 스무딩 0.023 / 3.79, jerk 0.044 / 15.1
        # → 부드러운 동작은 안정성 페널티 합 ≈ 0.9 (거리항의 ~12%), 급격한 동작은 ≈ 6.3 (거리항의 ~2배)
        # v2: 스무딩/jerk 0.1 → 3.0. v1 학습 결과에서 두 항목이 안정성 페널티 합의 1%에 불과해 사실상 미작동이었음
        #     (베이스 78%, 토크 21%). 3.0이면 v1 정책 기준 안정성 합의 약 20%를 차지
        if self.stability:
            reward -= 10.0 * torque_cost
            reward -= 1.5 * base_cost
            reward -= 3.0 * smooth_cost
            reward -= 3.0 * jerk_cost

        # 충돌 시 종료하지 않고 매 스텝 페널티만 부여 (충돌로 에피소드를 빨리 끝내는 꼼수 방지)
        in_contact = len(p.getContactPoints(bodyA=self.robot, bodyB=self.plane)) > 0
        if in_contact:
            reward -= 10.0

        # 정착 조건: 5mm 이내 + TCP 속도 2cm/s 미만일 때만 성공 (빠르게 통과하는 것은 성공 아님)
        ee_speed = np.linalg.norm(p.getLinkState(self.robot, self.ee_link_index, computeLinkVelocity=1)[6])
        terminated = False
        if distance < 0.005 and ee_speed < 0.02:
            terminated = True
            reward += 150.0

        truncated = self.step_cnt >= self.max_steps

        info = {"distance": distance, "ee_speed": ee_speed, "contact": in_contact, "success": terminated,
                "torque_cost": torque_cost, "base_cost": base_cost, "smooth_cost": smooth_cost, "jerk_cost": jerk_cost}
        return obs, reward, terminated, truncated, info
