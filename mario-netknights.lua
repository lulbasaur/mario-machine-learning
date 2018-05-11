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
--13*13 + 1 = 170
  INPUTS = INPUTSIZE+1
  OUTPUTS = #BUTTONNAMES
  HIDDENNODES = 20
  HIDDENLAYERS = 2
  LAYERS = HIDDENLAYERS + 2
  MAXNEURON = 1000000

  POPULATION_NR = 10
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
  return inputs
end
------------------MATH-----------------------------------------


function sigmoid(x)
	return 2/(1+math.exp(-4.9*x))-1
end

---------------------ANN--------------------------------------
function new_neuron()
  local neuron = {}
  neuron.value = 0.0
  return neuron
end


function new_weight_matrix(inn,outn)
  local weight_matrix = {}
  for i=1,inn do
    weight_matrix[i] = {}
    for j=1,outn do
      weight_matrix[i][j] = math.random()
    end
  end
  return weight_matrix
end

function new_neural_network()
  local neural_network = {}
  -- input neuron array
  neural_network.input_neurons = {}
  --hidden layer array of neuron arrays
  neural_network.hidden_layers = {}
  for i=1,HIDDENLAYERS do
    neural_network.hidden_layers[i] = {}
  end
  -- output neuron array
  neural_network.outputs = {}

  --weight array of arrays
  neural_network.weights = {}
  for i=1,LAYERS-1 do
    neural_network.weights[i] = {}
  end

  --INPUT NEURONS
  for i = 1,INPUTS do
    neural_network.input_neurons[i] = new_neuron()
  end

  --HIDDEN LAYERS
  for i=1,HIDDENLAYERS do
    for j=1,HIDDENNODES do
      neural_network.hidden_layers[i][j] = new_neuron()
    end
  end

  --OUTPUT NEURONS
  for i = 1,OUTPUTS do
    neural_network.outputs[i] = new_neuron()
  end

  --init weights hidden to output
  neural_network.weights[1] = new_weight_matrix(INPUTS,HIDDENNODES)
  neural_network.weights[2] = new_weight_matrix(HIDDENNODES,HIDDENNODES)
  neural_network.weights[3] = new_weight_matrix(HIDDENNODES,OUTPUTS)

  return neural_network
end

function evaluate_network(network,inputs)
  --inputs is from get_inputs
  --init inputs
  for i=1,INPUTS do
    network.input_neurons[i].value = inputs[i]
  end

  --update hidden layer values
  for i=1,HIDDENNODES do
    for j=1,INPUTS-1 do
      local current_in_value = network.input_neurons[j].value
      local current_weight_value = network.weights[1][j][i]
      local old = network.hidden_layers[1][i].value
      network.hidden_layers[1][i].value = old + current_in_value * current_weight_value
    end
    network.hidden_layers[1][i].value = sigmoid(network.hidden_layers[1][i].value)
  end

  --update hidden layer values
  for i=1,HIDDENNODES do --l2
    for j=1,HIDDENNODES do --l1
      local current_in_value = network.hidden_layers[1][j].value
      local current_weight_value = network.weights[2][j][i]
      local old = network.hidden_layers[2][i].value
      network.hidden_layers[2][i].value = old + current_in_value * current_weight_value
    end
    network.hidden_layers[2][i].value = sigmoid(network.hidden_layers[2][i].value)
  end

  --update hidden layer values
  for i=1,OUTPUTS do
    for j=1,HIDDENNODES do
      local current_in_value = network.hidden_layers[2][j].value
      local current_weight_value = network.weights[3][j][i]
      local old = network.outputs[i].value
      network.outputs[i].value = old + current_in_value * current_weight_value
    end
    network.outputs[i].value = sigmoid(network.outputs[i].value)
  end

  local outputs = {}
  for o=1,OUTPUTS do
    local button = "P1 " .. BUTTONNAMES[o]
    if network.input_neurons[o].value > 0 then
      outputs[button] = true
    else
      outputs[button] = false
    end
  end

  return outputs
end

function new_genome()
  local genome = {}
  genome.network = new_neural_network()
  genome.fitness = 0
  genome.rank = 0
  return genome
end

function new_population()
  local population = {}
  population.current_best = 1
  population.individuals = {}
  for i=1,POPULATION_NR do
    population.individuals[i] = new_genome()
  end
  return population
end



function clear_joypad()
	controller = {}
	for b = 1,#BUTTONNAMES do
		controller["P1 " .. BUTTONNAMES[b]] = false
	end
	joypad.set(controller)
end


local clock = os.clock
function sleep(n)  -- seconds
  local t0 = clock()
  while clock() - t0 <= n do end
end

function test()
  --sleep(1)
  local nn = new_neural_network()
  local current_inputs = get_inputs()
  local outputs = evaluate_network(nn,current_inputs)
  return outputs
end


function run_individual(indiv_index,pop)
  print("run_individual")
  local time_out_const = 40
  local time_out = time_out_const
  local rightmost = 0
  savestate.load(FILENAME);
  while time_out > 0 do
    local inputs = get_inputs()
    local outputs =  evaluate_network(pop.individuals[indiv_index].network,inputs)
    controller = outputs
    if controller["P1 Left"] and controller["P1 Right"] then
      controller["P1 Left"] = false
      controller["P1 Right"] = false
    end
    if controller["P1 Up"] and controller["P1 Down"] then
      controller["P1 Up"] = false
      controller["P1 Down"] = false
    end
    controller["P1 Right"] = true
    joypad.set(controller)
    get_positions()
    if mario_x > rightmost then
      rightmost = mario_x
      time_out = time_out_const
    end
    time_out = time_out - 1
    emu.frameadvance()
  end

end
--savestate.load(FILENAME);
population = new_population()
while true do
	clear_joypad()
  current_individual=1
  for i=1,POPULATION_NR do
    print("running individual:")
    print(i)
    run_individual(i,population)
  end

  console.writeline("newframe")
end
