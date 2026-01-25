import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data as pd
import numpy as np
import time
import os
from stable_baselines3 import SAC

# ==============================================================================
# 1. Franka Panda 고정밀 환경 (테스트용)
# ==============================================================================
class PandaHighPrecisionEnv(gym.Env):
    def __init__(self, render=True):
        super(PandaHighPrecisionEnv, self).__init__()
        
        self.render_mode = render
        if self.render_mode:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            # 카메라 시점 조정 (로봇과 타겟이 잘 보이도록)
            p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0.5, 0, 0.3])
        else:
            p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pd.getDataPath())
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        
        self.joint_indices = [0, 1, 2, 3, 4, 5, 6] 
        self.ee_link_index = 11                    
        self.ll = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89])
        self.ul = np.array([ 2.89,  1.76,  2.89, -0.06,  2.89,  3.75,  2.89])
        
        self.sim_step = 1./240.
        self.control_dt = 0.05 
        self.max_steps = 200   
        self.robot = None
        self.target_pos = None

    def _get_obs(self):
        joint_states = p.getJointStates(self.robot, self.joint_indices)
        q = np.array([state[0] for state in joint_states])
        dq = np.array([state[1] for state in joint_states])
        ee_pos = np.array(p.getLinkState(self.robot, self.ee_link_index)[0])
        rel_pos = self.target_pos - ee_pos
        return np.concatenate([q, dq, ee_pos, rel_pos]).astype(np.float32)

    def _sample_valid_target(self):
        """가동 범위 내 무작위 좌표 생성 (테스트용)"""
        # 테스트 시에는 조금 더 다양한 위치를 확인하기 위해 범위를 넓게 잡음
        for _ in range(50):
            target = np.array([np.random.uniform(0.3, 0.7), np.random.uniform(-0.5, 0.5), np.random.uniform(0.2, 0.7)])
            joint_poses = p.calculateInverseKinematics(self.robot, self.ee_link_index, target)
            for i, idx in enumerate(self.joint_indices):
                p.resetJointState(self.robot, idx, joint_poses[i])
            p.stepSimulation()
            actual_pos = p.getLinkState(self.robot, self.ee_link_index)[0]
            if np.linalg.norm(target - actual_pos) < 0.01:
                return target
        return np.array([0.5, 0, 0.4])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        self.target_pos = self._sample_valid_target()
        
        # 타겟 시각화: 5mm를 체감하기 위해 아주 작은 구체(반지름 0.005) 생성
        v_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.005, rgbaColor=[0, 1, 0, 0.9])
        p.createMultiBody(baseVisualShapeIndex=v_id, basePosition=self.target_pos)

        # 초기 자세 랜덤화
        for i in range(7):
            p.resetJointState(self.robot, self.joint_indices[i], np.random.uniform(self.ll[i]*0.1, self.ul[i]*0.1))
        
        self.step_cnt = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_cnt += 1
        # [중요] 학습 시 사용했던 정밀 제어 스케일(0.03) 유지
        real_action = action * 0.03
        current_q = [p.getJointState(self.robot, i)[0] for i in self.joint_indices]
        target_q = np.clip(current_q + real_action, self.ll, self.ul)
        p.setJointMotorControlArray(self.robot, self.joint_indices, p.POSITION_CONTROL, targetPositions=target_q)
        
        for _ in range(int(self.control_dt / self.sim_step)):
            p.stepSimulation()
        
        obs = self._get_obs()
        dist = np.linalg.norm(self.target_pos - obs[14:17])
        terminated = bool(dist < 0.005) # 5mm 도달 여부
        truncated = bool(self.step_cnt >= self.max_steps)
        
        return obs, 0, terminated, truncated, {"distance": dist}

# ==============================================================================
# 2. 테스트 실행 함수
# ==============================================================================
# ==============================================================================
# 2. 테스트 실행 함수 (모든 에피소드 슬로우 모션 적용)
# ==============================================================================
def run_high_precision_test(model_path):
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return

    print(f"📦 고정밀 모델 로드 중: {model_path}")
    env = PandaHighPrecisionEnv(render=True)
    model = SAC.load(model_path)
    
    num_tests = 50
    success_count = 0

    for ep in range(num_tests):
        obs, _ = env.reset()
        done = False
        step_idx = 0
        
        print(f"▶ 테스트 {ep+1}/{num_tests} 시작...", end="")
        
        while not done:
            # deterministic=True: 학습된대로 최적의 경로로 이동
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_idx += 1
            
            # [수정됨] 조건문(if ep == 0)을 삭제하여 모든 에피소드에 딜레이 적용
            # 0.03초 딜레이 = 사람 눈에 자연스러운 속도 (약 30 FPS 느낌)
            time.sleep(0.03) 
        
        final_dist = info['distance']
        is_success = final_dist < 0.005
        if is_success: success_count += 1
        
        status = "✅ 성공" if is_success else "❌ 실패"
        print(f" {status} | 오차: {final_dist*1000:.2f}mm | 스텝: {step_idx}")
        
        # 에피소드 사이에 1초 휴식 (결과 확인용)
        time.sleep(1.0)

    print("-" * 50)
    print(f"📊 최종 테스트 결과: 성공률 {success_count/num_tests*100:.1f}%")
    p.disconnect()

if __name__ == "__main__":
    # 파일명이 맞는지 꼭 확인하세요
    run_high_precision_test("panda_sac_50m_legend.zip")
