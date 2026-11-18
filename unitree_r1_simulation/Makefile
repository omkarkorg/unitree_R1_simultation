.PHONY: help setup certs build check check-quest sim sim-headless teleop record reset reset-all ros2 ros2-shell demo mock test test-isaac grasp-test test-kit train-loco export-loco eval-loco clean

help:
	@echo "One-time:  sudo make setup  |  make certs  |  make build  |  make check"
	@echo "Run:       make sim (terminal 1)  ->  make teleop (terminal 2)  ->  Quest 3 browser"
	@echo "Extras:    make record | make reset (objects back) | make ros2 | make ros2-shell | make demo | make mock"
	@echo "Quest:     make check-quest [QUEST=<headset IP>]  (the Quest cannot open the page)"
	@echo "Tests:     make test (no GPU) | make test-isaac (sim running) | make grasp-test (sim running) | make test-kit"
	@echo "Walking:   make train-loco (hours) -> make export-loco -> make eval-loco  (the trained leg controller)"

setup:        ; ./scripts/setup_host.sh
certs:        ; ./scripts/gen_certs.sh
build:        ; ./scripts/build.sh
check:        ; ./scripts/check_host.sh
check-quest:  ; ./scripts/check_quest.sh $(QUEST)
sim:          ; ./scripts/run_sim.sh
sim-headless: ; SIM_HEADLESS=1 ./scripts/run_sim.sh
teleop:       ; ./scripts/run_teleop.sh
record:       ; ./scripts/run_teleop.sh --record
reset:        ; ./scripts/reset_scene.sh
reset-all:    ; ./scripts/reset_scene.sh all
ros2:         ; ./scripts/run_ros2.sh
ros2-shell:   ; ./scripts/run_ros2.sh bash
demo:         ; ./scripts/run_ros2.sh ros2 run r1_ros2 arm_demo
mock:         ; ./scripts/run_mock_sim.sh
test:         ; ./scripts/test_stack.sh
test-isaac:   ; ./scripts/test_stack.sh --isaac
grasp-test:   ; ./scripts/grasp_test.sh
test-kit:     ; ./scripts/test_assembly.sh
train-loco:   ; ./scripts/train_loco.sh
export-loco:  ; ./scripts/train_loco.sh --export
eval-loco:    ; ./scripts/train_loco.sh --evaluate

clean:
	docker compose --profile teleop --profile ros2 --profile mock down --remove-orphans
	@echo "Images and Isaac Sim cache volumes are kept. Remove caches with:"
	@echo "  docker volume rm \$$(docker volume ls -q | grep '^unitree-r1_')"
