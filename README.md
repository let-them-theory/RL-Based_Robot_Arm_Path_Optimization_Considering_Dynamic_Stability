# 프로젝트 목표
동적 안정성을 고려하여 Python을 활용한 강화 학습을 통하여 7DOF Franka panda 로봇 암의 경로를 최적화


# 프로젝트 주제 선정 배경
본 프로젝트의 주제는 물체를 인식해서 물건을 잡는 AI 로봇 시연 과정에서 나타난 두 가지 문제점을 해결하기 위해 선정되었습니다.

1. 목표 지점이 명확하지 않다
- 물체를 인식했음에도 불구하고 '어떻게' 접근하고 잡아야 하는지에 대한 세부적인 Motion Plan이 부족하여, 실제 구동 시 손끝(End-effector)의 위치가 불분명해지는 현상이 발생했습니다.

2. 동역학적 불안정성
- 로봇 팔이 오직 '목표 도달'이라는 거리 최적화에만 과도하게 몰입한 결과, 자신의 물리적 상태(Dynamics)를 고려하지 않고 모든 관절이 급격하게 목표로 빨려 들어가는 현상을 보였습니다.

=> 이러한 연쇄적인 오차들이 발생하여 Base frame이 흔들리는 현상이 발생하였고 이에 따라 End-effector(이하 EE)에도 큰 영향을 주어 초기 목표에 도달했음에도 불구하고 정확도가 매우 떨어진다는 문제가 발생했습니다.

이러한 문제 상황을 개선해보고자하여 '동작 안정성을 고려한 강화학습기반 7축 로봇암의 경로 최적화' 프로젝트를 시작하게 되었습니다.

# 프로젝트 개요
7DOF를 가지는 Franka panda의 로봇 암이 임의의 점에 도달할 수 있는 경우의 수는 무수히 많습니다.

본 프로젝트는 로봇 암의 가동범위 내에 랜덤한 점을 생성하여 가동범위 내의 영역에 모두 도달 할 수 있도록 하였고,

EE가 효율적으로 목표점에 도달하며 동적 안정성과 에너지 효율성에 초점을 맞춘 최적 경로를 찾기 위해 노력하였습니다.

또한, 목표 도달 및 동적 안정성/에너지 효율성을 확보하기 위한 보상 체계는 크게 세 가지 요소인 (1) 다단계 거리 보상 (2) 각 조인트의 토크 최소화 및 가속도의 급격한 변화 제약 (3) 안전성 제약으로 학습하였습니다.

# 가동범위 정의
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


# 보상 체계
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

# 보상 함수
(1) 다단계 거리 보상 = 거리 보상 + 정밀 접근 보상
(2) 각 조인트의 토크 최소화 
(3) 가속도의 급격한 변화 발생 방지

        # 5. [보상 설계 - 매우 중요]
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



