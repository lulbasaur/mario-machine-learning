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
  HIDDENNODES = 30
  HIDDENLAYERS = 2
  LAYERS = HIDDENLAYERS + 2
  CROSSOVER_PROB = 0.75
  CROSSOVER_SPAN = 10
  CROSSOVER_THRESH = 10
  MUTATE_PROB = 0.75
  POPULATION_NR = 10
  WEIGHT_SPAN = 4
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



function new_weight_array(inn,outn)
  local weight_array = {}
  for i=1,inn do
    for j=1,outn do
      weight_array[i*outn + j] = math.random(-WEIGHT_SPAN,WEIGHT_SPAN)
      --weight_array[i*outn + j] = math.random()
    end
  end
  return weight_array
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
  neural_network.weights[1] = new_weight_array(INPUTS,HIDDENNODES)
  neural_network.weights[2] = new_weight_array(HIDDENNODES,HIDDENNODES)
  neural_network.weights[3] = new_weight_array(HIDDENNODES,OUTPUTS)
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
      local current_weight_value = network.weights[1][j*HIDDENNODES + i]
      local old = network.hidden_layers[1][i].value
      network.hidden_layers[1][i].value = old + current_in_value * current_weight_value
    end
    network.hidden_layers[1][i].value = sigmoid(network.hidden_layers[1][i].value)
  end

  --update hidden layer values
  for i=1,HIDDENNODES do --l2
    for j=1,HIDDENNODES do --l1
      local current_in_value = network.hidden_layers[1][j].value
      local current_weight_value = network.weights[2][j*HIDDENNODES+i]
      local old = network.hidden_layers[2][i].value
      network.hidden_layers[2][i].value = old + current_in_value * current_weight_value
    end
    network.hidden_layers[2][i].value = sigmoid(network.hidden_layers[2][i].value)
  end

  --update hidden layer values
  for i=1,OUTPUTS do
    for j=1,HIDDENNODES do
      local current_in_value = network.hidden_layers[2][j].value
      local current_weight_value = network.weights[3][j*OUTPUTS+i]
      local old = network.outputs[i].value
      network.outputs[i].value = old + current_in_value * current_weight_value
    end
    network.outputs[i].value = sigmoid(network.outputs[i].value)
  end

  local outputs = {}
  for o=1,OUTPUTS do
    local button = "P1 " .. BUTTONNAMES[o]
    if network.outputs[o].value > 0 then
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


function crossover(network1,network2)
  local new_network = new_neural_network()
  --nn1 weights
  local w1_1 = network1.weights[1]
  local w1_2 = network1.weights[2]
  local w1_3 = network1.weights[3]
  --nn2 weights
  local w2_1 = network2.weights[1]
  local w2_2 = network2.weights[2]
  local w2_3 = network2.weights[3]

  -- 170*20 * 3400 --- 850
  local span1 = (INPUTS*HIDDENNODES)/CROSSOVER_SPAN
  -- 20 * 20 = 400 --- 100
  local span2 = (HIDDENNODES*HIDDENNODES)/CROSSOVER_SPAN
  -- 20*4 = 80 --- 20
  local span3 = (HIDDENNODES*OUTPUTS)/CROSSOVER_SPAN

  local new_w1 = w1_1
  local new_w2 = w1_2
  local new_w3 = w1_3

  local start_index_1 = math.random(1, (INPUTS*HIDDENNODES)-span1)
  local start_index_2 = math.random(1, (HIDDENNODES*HIDDENNODES)-span2)
  local start_index_3 = math.random(1, (HIDDENNODES*OUTPUTS)-span3)

  for i=start_index_1,span1 do
    new_w1[i] = w2_1[i]
  end
  for i=start_index_2,span2 do
    new_w2[i] = w2_2[i]
  end
  for i=start_index_3,span3 do
    new_w3[i] = w2_3[i]
  end
  new_network.weights[1] = new_w1
  new_network.weights[2] = new_w2
  new_network.weights[3] = new_w3


  return new_network
end

function mutate(net)
  local w1 = net.weights[1]
  local w2 = net.weights[2]
  local w3 = net.weights[3]
  local index_1 = math.random(1, (INPUTS*HIDDENNODES))
  local index_2 = math.random(1, (HIDDENNODES*HIDDENNODES))
  local index_3 = math.random(1, (HIDDENNODES*OUTPUTS))
  w1[index_1] = math.random(-WEIGHT_SPAN,WEIGHT_SPAN)
  w2[index_2] = math.random(-WEIGHT_SPAN,WEIGHT_SPAN)
  w3[index_3] = math.random(-WEIGHT_SPAN,WEIGHT_SPAN)
  net.weights[1] = w1
  net.weights[2] = w2
  net.weights[3] = w3
  return net
end

function evolve(popu)
  if math.random() < CROSSOVER_PROB then
    local top1 = popu.individuals[1].network
    local top2 = popu.individuals[2].network
    for i=1,POPULATION_NR do
      if math.random() < MUTATE_PROB then
        print("mutate!")
        popu.individuals[i].network = mutate(popu.individuals[i].network)
      end
      if i>POPULATION_NR-(POPULATION_NR/CROSSOVER_THRESH) then
        popu.individuals[i].network = crossover(top1,top2)
      end
    end
  end

  return popu
end

function run_individual(indiv_index,pop)
  print("run_individual")
  local time_out_const = 20
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
    --controller["P1 Left"] = false
    joypad.set(controller)
    get_positions()
    if mario_x > rightmost then
      rightmost = mario_x
      time_out = time_out_const
    end
    time_out = time_out - 1
    pop.individuals[indiv_index].fitness = mario_x
    --print("fitness",pop.individuals[indiv_index].fitness )
    emu.frameadvance()
  end
end

function average_fitness(popula)
  local aver = 0
  for i=1,POPULATION_NR do
    aver = aver + popula.individuals[i].fitness
  end
  aver = aver/POPULATION_NR
  return aver
end

function average_fitness_increase_check(popul,aver, old_avr)
  if old_avr > aver then
    print("getting worse! improving population...")
    for i=POPULATION_NR/2,POPULATION_NR do
      popul.individuals[i].network = popul.individuals[1].network
    end
  end
  print("getting better! keep it up!")
  return popul
end

--savestate.load(FILENAME);
population = new_population()
old_average = 0
while true do
	clear_joypad()
  current_individual=1
  for i=1,POPULATION_NR do
    print("running individual:")
    print(i)
    run_individual(i,population)
  end
  table.sort(population.individuals, function (a,b)
    return (a.fitness > b.fitness)
  end)
  for i=1,POPULATION_NR do
    print("fitness", population.individuals[i].fitness)
  end
  population = evolve(population)
  average = average_fitness(population)
  population = average_fitness_increase_check(population,average,old_average)
  old_average = average
  console.writeline("newframe")
end
