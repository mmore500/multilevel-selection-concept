import uuid


def strong_uuid4_str() -> str:
    """
    Generate a “UUID-8” by:
      1) taking uuid1(), uuid3(), uuid4(), uuid5()
      2) XOR-ing all four 16-byte sequences together
      3) setting the version field to 8 and the variant to RFC-4122
    """
    # 1) generate the four source UUIDs
    u1 = uuid.uuid1()
    u3 = uuid.uuid3(uuid.NAMESPACE_DNS, u1.hex)
    u4 = uuid.uuid4()
    u5 = uuid.uuid5(uuid.NAMESPACE_DNS, u4.hex)

    # 2) XOR all their bytes together
    mixed = bytes(
        a ^ b ^ c ^ d
        for a, b, c, d in zip(u1.bytes, u3.bytes, u4.bytes, u5.bytes)
    )

    # 3) fix up version (4) and variant (RFC-4122) bits
    ba = bytearray(mixed)
    ba[6] = (ba[6] & 0x0F) | (8 << 4)  # set version to 4
    ba[8] = (ba[8] & 0x3F) | 0x80  # set variant to 10xxxxxx

    return str(uuid.UUID(bytes=bytes(ba)))
