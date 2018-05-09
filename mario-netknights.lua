Filename = "DP1.state"
  ButtonNames = {
    "A",
    "B",
    "Left",
    "Right",
  }

function getPositions()
	marioX = memory.read_s16_le(0x94)
	marioY = memory.read_s16_le(0x96)

	local layer1x = memory.read_s16_le(0x1A);
	local layer1y = memory.read_s16_le(0x1C);

	screenX = marioX-layer1x
	screenY = marioY-layer1y
end


while true do

  emu.frameadvance()
end
