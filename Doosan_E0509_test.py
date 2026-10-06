import pybullet as p
import time
import os
from stable_baselines3 import SAC

# 학습과 동일한 환경을 사용 (관측/제어/토크 한계가 학습 때와 어긋나지 않도록)
from Doosan_E0509_train import E0509Env

# ==============================================================================
# 테스트 실행 함수 (모든 에피소드 슬로우 모션 적용)
# ==============================================================================
def run_high_precision_test(model_path, num_tests=50):
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return

    print(f"📦 모델 로드 중: {model_path}")
    env = E0509Env(render=True, max_steps=200)
    model = SAC.load(model_path)

    success_count = 0
    contact_count = 0

    for ep in range(num_tests):
        obs, _ = env.reset()
        done = False
        step_idx = 0
        touched = False

        print(f"▶ 테스트 {ep+1}/{num_tests} 시작...", end="")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            touched = touched or info['contact']
            step_idx += 1
            time.sleep(0.03)

        final_dist = info['distance']
        is_success = info['success']  # 5mm 이내 + TCP 속도 2cm/s 미만 (정착)
        if is_success: success_count += 1
        if touched: contact_count += 1

        status = "✅ 성공" if is_success else "❌ 실패"
        print(f" {status} | 오차: {final_dist*1000:.2f}mm | 속도: {info['ee_speed']*1000:.1f}mm/s | 스텝: {step_idx}" + (" | ⚠️ 바닥 접촉" if touched else ""))

        time.sleep(1.0)

    print("-" * 50)
    print(f"📊 최종 테스트 결과: 성공률 {success_count/num_tests*100:.1f}% | 바닥 접촉 에피소드 {contact_count}/{num_tests}")
    p.disconnect()

if __name__ == "__main__":
    import sys
    # 사용법: python Doosan_E0509_test.py [모델 경로]  (기본: 안정성 페널티 모델 시드 1)
    run_high_precision_test(sys.argv[1] if len(sys.argv) > 1 else "e0509_2f85_sac_v2_s1.zip")
