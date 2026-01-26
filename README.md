# 1. 프로젝트 목표
동적 안정성을 고려하여 Python을 활용한 강화 학습을 통하여 7DOF Franka panda 로봇 암의 경로를 최적화


# 2. 프로젝트 주제 선정 배경
본 프로젝트의 주제는 물체를 인식해서 물건을 잡는 AI 로봇 시연 과정에서 나타난 두 가지 문제점을 해결하기 위해 선정되었습니다.

1. 목표 지점이 명확하지 않다
- 물체를 인식했음에도 불구하고 '어떻게' 접근하고 잡아야 하는지에 대한 세부적인 Motion Plan이 부족하여, 실제 구동 시 손끝(End-effector)의 위치가 불분명해지는 현상이 발생했습니다.

2. 동역학적 불안정성
- 로봇 팔이 오직 '목표 도달'이라는 거리 최적화에만 과도하게 몰입한 결과, 자신의 물리적 상태(Dynamics)를 고려하지 않고 모든 관절이 급격하게 목표로 빨려 들어가는 현상을 보였습니다.

=> 이러한 연쇄적인 오차들이 발생하여 Base frame이 흔들리는 현상이 발생하였고 이에 따라 End-effector(이하 EE)에도 큰 영향을 주어 초기 목표에 도달했음에도 불구하고 정확도가 매우 떨어진다는 문제가 발생했습니다.

이러한 문제 상황을 개선해보고자하여 '동작 안정성을 고려한 강화학습기반 7축 로봇암의 경로 최적화' 프로젝트를 시작하게 되었습니다.

# 3. 프로젝트 개요
7DOF를 가지는 Franka panda의 로봇 암이 임의의 점에 도달할 수 있는 경우의 수는 무수히 많습니다.

본 프로젝트는 로봇 암의 가동범위 내에 랜덤한 점을 생성하여 가동범위 내의 영역에 모두 도달 할 수 있도록 하였고,

EE가 효율적으로 목표점에 도달하며 동적 안정성과 에너지 효율성에 초점을 맞춘 최적 경로를 찾기 위해 노력하였습니다.

또한, 목표 도달 및 동적 안정성/에너지 효율성을 확보하기 위한 보상 체계는 크게 세 가지 요소인 (1) 다단계 거리 보상 (2) 각 조인트의 토크 최소화 및 가속도의 급격한 변화 제약 (3) 안전성 제약으로 학습하였습니다.

# 4. 프로젝트 구조

      [ 1. 문제 정의 (Problem) ]
            |
            | "7자유도의 무한한 해 & 동적 불안정성 해결"
            v
      [ 2. 환경 구축 (Environment) ]
            |
            +-- 시뮬레이터: PyBullet (Franka Panda)
            +-- 최적화 기술: IK 기반 고속 리셋 (FPS 50배 향상)
            |
            v
      [ 3. 강화학습 (RL Training) ] <--------+
            |                                |
            +-- 알고리즘: SAC (Soft Actor-Critic)
            +-- 보상함수: 거리 + 에너지 최소화 + 진동 억제
            +-- 데이터: Replay Buffer (Off-Policy)
            |                                |
            +--------------------------------+ (Feedback Loop)
            |
            v
      [ 4. 성능 평가 (Evaluation) ]
            |
            +-- 성공률: 95% 달성 (오차 < 5mm)
            +-- 안정성: Base Frame 진동 제어 확인
            |
            v
      [ 5. 결론 (Conclusion) ]
            |
            "Fast & Stable: 고속 정밀 파지 시스템 완성"

# 5. 학습 환경 구축
gymnasium공간에 pybullet을 사용하여 Franka panda를 불러왔습니다.

학습 알고리즘은 SAC 방식을 사용하여 거리보상 + 엔트로피를 높여 무한한 해 공간을 다양하게 탐험합니다. 

그리고 '에너지 최소화 보상 함수'를 통해, 무한한 자세들 중 가장 에너지를 적게 쓰고 진동이 없는 최적의 자세를 스스로 찾아내도록 학습시켰습니다.

        import gymnasium as gym
        from gymnasium import spaces
        import pybullet as p
        import pybullet_data as pd
        import numpy as np
        import time
        import os
        
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
        from stable_baselines3.common.env_util import make_vec_env


# 5-1 라이브러리 실행을 위한 패키지파일 설치
       
        pip install gymnasium
        pip install pybullet 
        pip install stable-baselines3
        pip install shimmy
        pip install torch torchvision
        
vs_Community에서 c++을 활용한 데스크톱 개발항목 체크해서 설치

https://download.visualstudio.microsoft.com/download/pr/64dbe648-9527-4b8e-9b08-04d2228e1191/a3f10a06535a1a22678db2a5561b8599f42687c68beec08ac560a197e2901980/vs_Community.exe



# 6. 가동범위 정의
Richable Workspace 정의
-> Rank(A)=n
-> det(A) ≠ 0
-> A의 역행렬 A^-1이 존재할 때
=> 특이점이 아닌 상태를 만족하기 위해 다음과 같은 한계점을 정의하였습니다.

        # 2. 행동(Action)과 관측(Observation) 공간 정의
        # 행동: 7개 관절의 속도/위치 변화량 (-1.0 ~ 1.0)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        # 관측: 20개 데이터 (관절각도 7 + 속도 7 + 손끝위치 3 + 목표상대위치 3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        
        # 3. 로봇 하드웨어 설정 (관절 인덱스 및 한계값)
        self.joint_indices = [0, 1, 2, 3, 4, 5, 6] 
        self.ee_link_index = 11 # Panda 로봇의 손끝(End-Effector) 링크 번호
        
        # 관절 각도 하한(ll)과 상한(ul) - 로봇이 꺾일 수 있는 물리적 한계
        self.ll = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89])
        self.ul = np.array([ 2.89,  1.76,  2.89, -0.06,  2.89,  3.75,  2.89])
        self.joint_range = self.ul - self.ll


# 7. 보상 체계
행동을 정의하고 조인트가 스스로 자신의 상태를 파악하고 물리 엔진을 기반으로 계산하여 판단 할 수 있도록 함


        def step(self, action):
        # 1. 행동 스케일링 (너무 급격하게 움직이지 않도록 0.03 곱함)
        real_action = action * 0.03 
        
        # 2. 모터 제어 명령 (Position Control)
        # 현재 각도 + 행동(변화량) = 목표 각도
        p.setJointMotorControlArray(..., targetPositions=target_q, forces=[500]*7)
        
        # 3. 물리 엔진 업데이트 (0.05초 동안 시뮬레이션 진행)
        for _ in range(int(self.control_dt / self.sim_step)):
            p.stepSimulation()

        # 4. 관측 및 거리 계산
        obs = self._get_obs()
        distance = np.linalg.norm(self.target_pos - ee_pos)

# 8. 보상 함수
(1) 다단계 거리 보상 = 거리 보상 + 정밀 접근 보상
(2) 각 조인트의 토크 최소화 
(3) 가속도의 급격한 변화 발생 방지

        # 5. [보상 설계]
        # (1) 거리 보상: 가까울수록 점수가 덜 깎임 (음수 보상)
        reward = -distance * 20.0 
        
        # (2) 정밀 접근 보상: 아주 가까워지면 추가 점수
        if distance < 0.02:
            reward += (0.02 - distance) * 200.0 
        
        # (3) 에너지 패널티: 힘(Torque)을 많이 쓰거나, 급격히 움직이면 감점
        reward -= 0.01 * np.mean(np.square(self.current_torques))
        reward -= 0.1 * np.mean(np.square(action - self.prev_action))
        
        # 6. 종료 조건 (성공 또는 충돌)
        if len(contact_points) > 0: # 충돌 시
            reward -= 10.0
            terminated = True
        elif distance < 0.005: # 목표 도달 성공 시
            terminated = True
            reward += 150.0 # 큰 성공 보상
        
        return obs, reward, terminated, truncated, {"distance": distance}

# 9. 학습 과정
학습은 SAC 알고리즘으로 코어 14개를 사용하여 cmd 환경에서 직접 병렬연산으로 1000만회 학습하였습니다. 
        
        if __name__ == "__main__":
            NUM_CPU = 14  # CPU 코어 사용 개수
            TOTAL_TIMESTEPS = 10000000 # 총 학습 횟수 (1천만 회)
            
            # ... (중략)
        
            # 1. 벡터화 환경 생성 (병렬 처리)
            env = make_vec_env(
                PandaLongRunEnv, 
                n_envs=NUM_CPU, 
                vec_env_cls=SubprocVecEnv, # 멀티프로세싱 사용
                env_kwargs={'render': False} 
            )
            env = VecMonitor(env) # 학습 로그 기록용 래퍼
        
            # 2. SAC 모델 정의
            model = SAC(
                "MlpPolicy", # 이미지(CNN)가 아닌 수치 데이터(MLP) 사용
                env, 
                verbose=1,
                learning_rate=2e-4, # 학습률
                batch_size=512,     # 한 번에 학습할 데이터 양
                tensorboard_log="./panda_50m_logs/" # 로그 저장 경로
            )
        
            # 3. 학습 시작
            model.learn(total_timesteps=TOTAL_TIMESTEPS)
            
            # 4. 저장
            model.save(MODEL_NAME)


# 10. 프로젝트 결과 분석
- 동적 보상이 없이 거리 보상으로만 학습했을 때

https://youtu.be/T843mICs6Xk

https://github.com/user-attachments/assets/96da4a5b-14f0-4ee9-944c-526123f9346c




- 동적 안정성을 고려하여 학습했을 때

https://youtu.be/gy8SVOMf_HE

https://github.com/user-attachments/assets/0634166a-76bd-48bd-89d1-b83c7f732a45



두 영상을 보았을 때 육안으로도 쉽게 구분할 수 있을 정도로 각 조인트의 움직임이 부드러워졌고 정밀도가 확연하게 차이남을 알 수 있습니다.

# 11. 결론 및 고찰
최종 학습된 결과를 테스트 코드를 통해 분석하면 목표지점에 5mm이내로 95% 정확도로 접근하는 것을 알 수 있습니다.

또한, 나머지 시행들은 약 1cm 이내로 접근하며 좋은 정확도를 가진다고 판단할 수 있습니다.


프로젝트를 진행하며 대략 3억번 정도의 학습을 진행하였고 강화학습이 어떤 것 인지, 보상 함수 설계가 얼마나 중요한지 알 수 있게 되었습니다.

그러나 좋은 정확도를 가진 모델을 학습시켰지만, 이 모델이 역학적으로 안정적인 자세로 목표점에 도달했는가에 대한 의문은 해소하지 못했습니다.

다음 프로젝트 주제는 이번에 학습한 모델이 역학적으로 안정적인 모델인가에 대한 주제로 진행하려고 합니다.


#12 다음 프로젝트 개요
프로젝트의 초기 설계는 아래와 같습니다.

1. EE에 x, y, z 방향에 랜덤하게 input을 가함
- 안정성, 안전성을 테스트하기 위해 Impulse
- 반응성을 테스트하기 위해 Unit step
- 추종성을 테스트하기 위해 Ramp
- 내구성을 테스트하기 위해 Harmonic

=> 입력한 힘의 주파수 대비 EE의 흔들림을 Bode Plot을 그려서 확인, settling time, overshoot, Steady-State오차 등을 분석하여 동적 거동 실험을 진행

