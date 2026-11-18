import struct

_HG_LOWCMD_FMT = "<2B2x" + "B3x5fI" * 35 + "5I"
_POLY = 0x04C11DB7


def crc32_core(words):
    crc = 0xFFFFFFFF
    for data in words:
        xbit = 0x80000000
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ _POLY
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if data & xbit:
                crc ^= _POLY
            xbit >>= 1
    return crc


def hg_lowcmd_crc(msg) -> int:
    values = [int(msg.mode_pr), int(msg.mode_machine)]
    for i in range(35):
        c = msg.motor_cmd[i]
        values += [int(c.mode), float(c.q), float(c.dq), float(c.tau), float(c.kp), float(c.kd), int(c.reserve)]
    values += [int(v) for v in msg.reserve[:4]]
    values.append(0)
    raw = struct.pack(_HG_LOWCMD_FMT, *values)
    words = struct.unpack("<%dI" % (len(raw) // 4), raw)
    return crc32_core(words[:-1])
