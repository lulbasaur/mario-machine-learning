-- Bra bok
-- www.cs.uni.edu/~schafer/4620/readings/Ai%20Techniques%20For%20Game%20Programming.pdf
FILENAME = "DP1.state"
  BUTTONNAMES = {
    "A",
    "B",
    "Left",
    "Right",
  }

  BOXRADIUS = 6
  INPUTSIZE = (BOXRADIUS*2+1)*(BOXRADIUS*2+1)

  INPUTS = INPUTSIZE+1
  OUTPUTS = #BUTTONNAMES
  MAXNEURON = 1000000
---------------------------- INPUT------------------------------
function get_positions()
	mario_x = memory.read_s16_le(0x94)
	mario_y = memory.read_s16_le(0x96)
	local layer_1_x = memory.read_s16_le(0x1A);
	local layer_1_y = memory.read_s16_le(0x1C);
	screen_x = mario_x-layer_1_x
	screen_y = mario_y-layer_1_y
end

function get_tile(dx, dy)
	x = math.floor((mario_x+dx+8)/16)
	y = math.floor((mario_y+dy)/16)

	return memory.readbyte(0x1C800 + math.floor(x/0x10)*0x1B0 + y*0x10 + x%0x10)
end

function get_sprites()
	local sprites = {}
	for slot=0,11 do
		local status = memory.readbyte(0x14C8+slot)
		if status ~= 0 then
			sprite_x = memory.readbyte(0xE4+slot) + memory.readbyte(0x14E0+slot)*256
			sprite_y = memory.readbyte(0xD8+slot) + memory.readbyte(0x14D4+slot)*256
			sprites[#sprites+1] = {["x"]=sprite_x, ["y"]=sprite_y}
		end
	end
	return sprites
end

function get_extended_sprites()
	local extended = {}
	for slot=0,11 do
		local number = memory.readbyte(0x170B+slot)
		if number ~= 0 then
			sprite_x = memory.readbyte(0x171F+slot) + memory.readbyte(0x1733+slot)*256
			sprite_y = memory.readbyte(0x1715+slot) + memory.readbyte(0x1729+slot)*256
			extended[#extended+1] = {["x"]=sprite_x, ["y"]=sprite_y}
		end
	end
	return extended
end

function get_inputs()
	get_positions()

	sprites = get_sprites()
	extended = get_extended_sprites()

	local inputs = {}

	for dy=-BOXRADIUS*16,BOXRADIUS*16,16 do
		for dx=-BOXRADIUS*16,BOXRADIUS*16,16 do
			inputs[#inputs+1] = 0

			tile = get_tile(dx, dy)
			if tile == 1 and mario_y+dy < 0x1B0 then
				inputs[#inputs] = 1
			end

			for i = 1,#sprites do
				distx = math.abs(sprites[i]["x"] - (mario_x+dx))
				disty = math.abs(sprites[i]["y"] - (mario_y+dy))
				if distx <= 8 and disty <= 8 then
					inputs[#inputs] = -1
				end
			end

			for i = 1,#extended do
				distx = math.abs(extended[i]["x"] - (mario_x+dx))
				disty = math.abs(extended[i]["y"] - (mario_y+dy))
				if distx < 8 and disty < 8 then
					inputs[#inputs] = -1
				end
			end
		end
	end


--poppe
function new_neuron()
  local neuron = {}
  neuron.connections = {}
  neuron.weight = 0.0
  return neuron
--create neuron here and return
end
--poppe


function new_neural_network()
  local neural_network = {}
  neural_network.neurons = {}

  for i = 1, Inputs do
    neural_network.neurons[i] = new_neuron()
  end

  for i = 1, Outputs do
    network.neurons[MaxNeuron+i] = new_neuron()
  end
--create neural network :)
end



while true do

  emu.frameadvance()
end
