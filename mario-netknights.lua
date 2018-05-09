

-- set buttons and file name
if gameinfo.getromname() == "Super Mario World (USA)" then
  Filename = "DP1.state"
  ButtonNames = {
    "A",
    "B",
    "Up",
    "Down",
    "Left",
    "Right",
  }
end

function get_positions()
	if gameinfo.getromname() == "Super Mario World (USA)" then
		marioX = memory.read_s16_le(0x94)
		marioY = memory.read_s16_le(0x96)

		local layer1x = memory.read_s16_le(0x1A);
		local layer1y = memory.read_s16_le(0x1C);

		screenX = marioX-layer1x
		screenY = marioY-layer1y
  end
end




while true do

  emu.frameadvance()
end
