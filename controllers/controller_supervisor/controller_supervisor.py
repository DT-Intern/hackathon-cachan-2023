from controller import Supervisor
from time import sleep

supervisor = Supervisor()
receiver = supervisor.getDevice("receiver")
receiver.enable(64)
emitter = supervisor.getDevice("emitter")

tt_02 = supervisor.getFromDef("TT02")
tt_02_translation = tt_02.getField("translation")
tt_02_rotation = tt_02.getField("rotation")

initial_translation = tt_02_translation.getSFVec3f()
initial_rotation = tt_02_rotation.getSFRotation()

# Transmit all checkpoints
while supervisor.step(int(supervisor.getBasicTimeStep())) != -1:
    # Receive all incoming messages
    if receiver.getQueueLength() > 0:
        message = receiver.getData().decode("utf-8")
        if message == "reset":
            tt_02_translation.setSFVec3f(initial_translation)
            tt_02_rotation.setSFRotation(initial_rotation)
            # tt_02_translation.setSFVec3f([-2, -2.5, 0.1])
            # tt_02_rotation.setSFRotation([0, 0, 1, 0])
        elif message == "checkpoints":
            checkpoints_data = []
            emitter.send(f"c{0}, {initial_translation[0]}, {initial_translation[1]}, {initial_translation[2]}".encode("utf-8"))
            for i in range(10):
                optional_checkpoint = supervisor.getFromDef(f"checkpoint{str(i)}")
                if optional_checkpoint is not None:
                    checkpoint_position = optional_checkpoint.getPosition()
                    emitter.send(
                        f"c{str(i + 1)}, {checkpoint_position[0]}, {checkpoint_position[1]}, {checkpoint_position[2]}".encode(
                            "utf-8"))
        receiver.nextPacket()

    position = tt_02.getPosition()
    emitter.send(f"p{position[0]}, {position[1]}, {position[2]}".encode("utf-8"))