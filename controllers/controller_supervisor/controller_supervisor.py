from controller import Supervisor

supervisor = Supervisor()
receiver = supervisor.getDevice("receiver")
receiver.enable(64)

tt_02 = supervisor.getFromDef("TT02")
tt_02_translation = tt_02.getField("translation")

while supervisor.step(int(supervisor.getBasicTimeStep())) != -1:
    if receiver.getQueueLength() > 0:
        message = receiver.getData().decode("utf-8")
        if message == "reset":
            tt_02_translation.setSFVec3f([-2, -2.5, 0.1])
        receiver.nextPacket()