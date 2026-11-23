from RsInstrument import RsInstrument
scope = RsInstrument("USB0::0x0AAD::0x01D6::205882::INSTR")
scope.write("SING")
scope.query("TRIGger:A:STATus?")