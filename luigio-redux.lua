--FILENAME = "DP1.state"
FILENAME = "smw1.state"
  BUTTONNAMES = {
    "A",
    "B",
    "Left",
    "Right",
  }

  BOXRADIUS = 6
  INPUTSIZE = (BOXRADIUS*2+1)*(BOXRADIUS*2+1)
  --ANN CONFIG
  INPUTS = INPUTSIZE+1
  OUTPUTS = #BUTTONNAMES
  HIDDENNODES = 10
  HIDDENLAYERS = 2
  LAYERS = HIDDENLAYERS + 2
  --GA CONFIG
  POPULATION_NR = 100 --at least 10


  --evolve parameters
  CROSSOVER_PROB_DEFAULT = 0.60
  NR_OF_PARENTS_TO_BREED_FROM = math.floor(POPULATION_NR*0.1)
  if NR_OF_PARENTS_TO_BREED_FROM < 2 then
    NR_OF_PARENTS_TO_BREED_FROM = 2
  end

  --crossover parameters
  CROSSOVER_SECTION_1 = 0.1 --n*m * CROSSOVER_SECTION_1 = span
  CROSSOVER_SECTION_2 = 0.1 --n*m * CROSSOVER_SECTION_2 = span
  CROSSOVER_SECTION_3 = 0.1 --n*m * CROSSOVER_SECTION_3 = span

  --crossover_2 parameters
  CROSSOVER_2_STRIDE = 3 --math.random(1,CROSSOVER_2_STRIDE)==1
  --mutate parameters
  MUTATE_RANK_RATIO = 0.1
  MUTATE_WEIGHT_PROCENT = 0.1
  --STALE_FITNESS_MULIPLIER = 5

  WEIGHT_LAYER_NR_1 = INPUTS*HIDDENNODES
  WEIGHT_LAYER_NR_2 = HIDDENNODES*HIDDENNODES
  WEIGHT_LAYER_NR_3 = HIDDENNODES*OUTPUTS


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
        --weight_array[i*outn + j] = math.random()*4-2
        weight_array[(i-1)*outn + j] = 0
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
    -- for i=1,N do
    --  for j=1,M do
    --    mt[i*M + j] = 0
    --update hidden layer values
    for i=1,HIDDENNODES do
      for j=1,INPUTS-1 do --WHY -1???????????????????
        local current_in_value = network.input_neurons[j].value
        local current_weight_value = network.weights[1][(j-1)*HIDDENNODES+i]
        local old = network.hidden_layers[1][i].value
        network.hidden_layers[1][i].value = old + (current_in_value * current_weight_value)
      end
      network.hidden_layers[1][i].value = sigmoid(network.hidden_layers[1][i].value)
      --print(network.hidden_layers[1][i].value)
    end
    --update hidden layer values
    for i=1,HIDDENNODES do --l2
      for j=1,HIDDENNODES do --l1
        local current_in_value = network.hidden_layers[1][j].value
        local current_weight_value = network.weights[2][(j-1)*HIDDENNODES+i]
        local old = network.hidden_layers[2][i].value
        network.hidden_layers[2][i].value = old + (current_in_value * current_weight_value)
      end
      network.hidden_layers[2][i].value = sigmoid(network.hidden_layers[2][i].value)
      --print(network.hidden_layers[2][i].value)
    end
    --update hidden layer values
    for i=1,OUTPUTS do
      for j=1,HIDDENNODES do
        local current_in_value = network.hidden_layers[2][j].value
        local current_weight_value = network.weights[3][(j-1)*OUTPUTS+i]
        local old = network.outputs[i].value
        --local old = 0
        network.outputs[i].value = old + (current_in_value * current_weight_value)
      end
      network.outputs[i].value = sigmoid(network.outputs[i].value)
      --print(network.outputs[i].value)
    end
    --set outputs
    local outputs = {}
    for o=1,OUTPUTS do
      local button = "P1 " .. BUTTONNAMES[o]
      if network.outputs[o].value > 0 then
        outputs[button] = true
      else
        outputs[button] = false
      end
    end
    --reset values
    for i=1,#network.hidden_layers[1] do
      network.hidden_layers[1][i].value =0.0
      network.hidden_layers[2][i].value =0.0
    end
    for i=1,#network.outputs do
      network.outputs[i].value = 0.0
    end
    --return the current outputs
    return outputs
  end
  ----------------------GA---------------------------------------
  function new_genome()
    local genome = {}
    genome.network = new_neural_network()
    genome.fitness = 1
    genome.prev_fitness = 1
    genome.same_fitness_count = 0
    genome.id = "name not set"
    genome.rank = POPULATION_NR
    genome.mutate_prob = 1
    return genome
  end

  function new_population()
    local population = {}
    population.current_best = 1
    population.individuals = {}
    for i=1,POPULATION_NR do
      population.individuals[i] = new_genome()
      population.individuals[i].id ="id_" .. i
    end
    return population
  end

  function weight_sum(net_to_calc)
    local sum = 0
    for i=1,#net_to_calc.weights[1] do
      sum = net_to_calc.weights[1][i] + sum
    end
    for i=1,#net_to_calc.weights[2] do
      sum = net_to_calc.weights[2][i] + sum
    end
    for i=1,#net_to_calc.weights[3] do
      sum = net_to_calc.weights[3][i] + sum
    end
    return sum
  end

  function mutate(geno_to_mut)
    local mutprob = (geno_to_mut.rank+(POPULATION_NR*MUTATE_RANK_RATIO))/POPULATION_NR
    geno_to_mut.mutate_prob = mutprob
    if math.random() < mutprob then
      local sfc = geno_to_mut.same_fitness_count
      local iter_nr_1 = math.floor(MUTATE_WEIGHT_PROCENT*(INPUTS*HIDDENNODES)) + (sfc * sfc)
      local iter_nr_2 = math.floor(MUTATE_WEIGHT_PROCENT*(HIDDENNODES*HIDDENNODES)) + (sfc * sfc)
      local iter_nr_3 = math.floor(MUTATE_WEIGHT_PROCENT*(HIDDENNODES*OUTPUTS)) + (sfc * sfc)
      if iter_nr_1 < 1 then
        iter_nr_1 = 1
      end
      if iter_nr_2 < 1 then
        iter_nr_2 = 1
      end
      if iter_nr_3 < 1 then
        iter_nr_3 = 1
      end
      --print("m1 mutating rank:" .. geno_to_mut.rank ..", " .. geno_to_mut.id ..  ", prob: " .. mutprob)
      --print("iterationes:" ..iter_nr_1 ..", " .. iter_nr_2 ..  ", " .. iter_nr_3)
      for i=1,iter_nr_1 do
        local index_1 = math.random(1, (INPUTS*HIDDENNODES))
        geno_to_mut.network.weights[1][index_1] = math.random()*4-2
      end
      for i=1,iter_nr_2 do
        local index_2 = math.random(1, (HIDDENNODES*HIDDENNODES))
        geno_to_mut.network.weights[2][index_2] = math.random()*4-2
      end
      for i=1,iter_nr_3 do
        local index_3 = math.random(1, (HIDDENNODES*OUTPUTS))
        geno_to_mut.network.weights[3][index_3] = math.random()*4-2
      end
    end
    return geno_to_mut
  end

  function mutate_2(geno_to_mut)
    local mutprob = (geno_to_mut.rank+(POPULATION_NR*MUTATE_RANK_RATIO))/POPULATION_NR
    geno_to_mut.mutate_prob = mutprob
    if math.random() < mutprob then
      local sfc = geno_to_mut.same_fitness_count
      local iter_nr_1 = math.floor(math.random()*(INPUTS*HIDDENNODES)) + (sfc * sfc)
      local iter_nr_2 = math.floor(math.random()*(HIDDENNODES*HIDDENNODES)) + (sfc * sfc)
      local iter_nr_3 = math.floor(math.random()*(HIDDENNODES*OUTPUTS)) + (sfc * sfc)
      if iter_nr_1 < 1 then
        iter_nr_1 = 1
      end
      if iter_nr_2 < 1 then
        iter_nr_2 = 1
      end
      if iter_nr_3 < 1 then
        iter_nr_3 = 1
      end
      --print("m2 mutating rank:" .. geno_to_mut.rank ..", " .. geno_to_mut.id ..  ", prob: " .. mutprob)
      --print("iterationes:" ..iter_nr_1 ..", " .. iter_nr_2 ..  ", " .. iter_nr_3)
      for i=1,iter_nr_1 do
        --math.randomseed(i+os.clock())
        local index_1 = math.random(1, (INPUTS*HIDDENNODES))
        --math.randomseed(i+i+os.clock())
        geno_to_mut.network.weights[1][index_1] = math.random()*4-2
      end
      for i=1,iter_nr_2 do
        --math.randomseed(i+i+os.clock())
        local index_2 = math.random(1, (HIDDENNODES*HIDDENNODES))
        --math.randomseed(i+i+i+os.clock())
        geno_to_mut.network.weights[2][index_2] = math.random()*4-2
      end
      for i=1,iter_nr_3 do
        --math.randomseed(i+i+i+i+os.clock())
        local index_3 = math.random(1, (HIDDENNODES*OUTPUTS))
        --math.randomseed(i+i+i+i+i+os.clock())
        geno_to_mut.network.weights[3][index_3] = math.random()*4-2
      end
    end
    return geno_to_mut
  end

  function crossover(genome_1_net,genome_2_net)
    local new_g_net = deep_copy_network_weights(genome_1_net)
    --local new_g_net = deepcopy(genome_1_net)
    local span1 = math.floor((INPUTS*HIDDENNODES)*CROSSOVER_SECTION_1)
    local span2 = math.floor((HIDDENNODES*HIDDENNODES)*CROSSOVER_SECTION_2)
    local span3 = math.floor((HIDDENNODES*OUTPUTS)*CROSSOVER_SECTION_3)
    local start_index_1 = math.random(1, (INPUTS*HIDDENNODES)-span1)
    local start_index_2 = math.random(1, (HIDDENNODES*HIDDENNODES)-span2)
    local start_index_3 = math.random(1, (HIDDENNODES*OUTPUTS)-span3)
    if span1 < 1 then
      span1 = 1
    end
    if span2 < 1 then
      span2 = 1
    end
    if span3 < 1 then
      span3 = 1
    end
    --print("index",{start_index_1,start_index_2,start_index_3})
    for i=start_index_1,span1 do
      new_g_net.weights[1][i] = genome_2_net.weights[1][i]
    end
    for i=start_index_2,span2 do
      new_g_net.weights[2][i] = genome_2_net.weights[2][i]
    end
    for i=start_index_3,span3 do
      new_g_net.weights[3][i] = genome_2_net.weights[3][i]
    end
    return new_g_net
  end

  function crossover_2(genome_1_net,genome_2_net)
    local new_g_net = deep_copy_network_weights(genome_1_net)
    for i=1,#genome_2_net.weights[1] do
      if math.random(1,CROSSOVER_2_STRIDE)==1 then
        new_g_net.weights[1][i] = genome_2_net.weights[1][i]
      end
    end
    for i=1,#genome_2_net.weights[2] do
      if math.random(1,CROSSOVER_2_STRIDE)==1 then
        new_g_net.weights[2][i] = genome_2_net.weights[2][i]
      end
    end
    for i=1,#genome_2_net.weights[3] do
      if math.random(1,CROSSOVER_2_STRIDE)==1 then
        new_g_net.weights[3][i] = genome_2_net.weights[3][i]
      end
    end
    return new_g_net
  end

  function evolve_2(pop_evolve_2)
    --crossover
    for i=NR_OF_PARENTS_TO_BREED_FROM+1,POPULATION_NR do
      if math.random()<CROSSOVER_PROB_DEFAULT then
        local id1 = math.random(1,NR_OF_PARENTS_TO_BREED_FROM)
        local id2 = math.random(1,NR_OF_PARENTS_TO_BREED_FROM)
        while id1==id2 do
          id1=math.random(1,NR_OF_PARENTS_TO_BREED_FROM)
          math.randomseed(i+os.clock())
          id2=math.random(1,NR_OF_PARENTS_TO_BREED_FROM)
        end
        --print("crossover: " .. pop_evolve.individuals[i].rank .. "<--" .. pop_evolve.individuals[id1].rank .. ", " .. pop_evolve.individuals[id2].rank)
        local cross_choice = math.random(1, 4)
        if cross_choice == 1 then
          pop_evolve_2.individuals[i].network = crossover(pop_evolve_2.individuals[id1].network,pop_evolve_2.individuals[i].network)
        elseif cross_choice == 2 then
          pop_evolve_2.individuals[i].network = crossover(pop_evolve_2.individuals[id1].network,pop_evolve_2.individuals[id2].network)
        elseif cross_choice == 3 then
          pop_evolve_2.individuals[i].network = crossover_2(pop_evolve_2.individuals[id1].network,pop_evolve_2.individuals[i].network)
        else
          pop_evolve_2.individuals[i].network = crossover_2(pop_evolve_2.individuals[id1].network,pop_evolve_2.individuals[id2].network)
        end
      end
    end
    --clone best to last position
    pop_evolve_2.individuals[POPULATION_NR].network = deep_copy_network_weights(pop_evolve_2.individuals[1].network)
    --mutate but skip best
    for i=2,POPULATION_NR do
      if math.random(1, 2)==1 then
        pop_evolve_2.individuals[i] = mutate(pop_evolve_2.individuals[i])
      else
        pop_evolve_2.individuals[i] = mutate_2(pop_evolve_2.individuals[i])
      end
    end
    return pop_evolve_2
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

  function deepcopy(orig)
      local orig_type = type(orig)
      local copy
      if orig_type == 'table' then
          copy = {}
          for orig_key, orig_value in next, orig, nil do
              copy[deepcopy(orig_key)] = deepcopy(orig_value)
          end
          setmetatable(copy, deepcopy(getmetatable(orig)))
      else -- number, string, boolean, etc
          copy = orig
      end
      return copy
  end

  function deep_copy_network_weights(network_orig)
    local new_net = new_neural_network()
    --for i=1,INPUTS do
    --  new_net.input_neurons[i].value = network_orig.input_neurons[i].value
    --end
    --for i=1,HIDDENLAYERS do
    --  for j=1,HIDDENNODES do
    --    new_net.hidden_layers[i][j].value = network_orig.hidden_layers[i][j].value
    --  end
    --end
    --for i = 1,OUTPUTS do
    --  new_net.outputs[i].value = network_orig.outputs[i].value
    --end
    for i=1,#network_orig.weights[1] do
      new_net.weights[1][i] = network_orig.weights[1][i]
    end
    for i=1,#network_orig.weights[2] do
      new_net.weights[2][i] = network_orig.weights[2][i]
    end
    for i=1,#network_orig.weights[3] do
      new_net.weights[3][i] = network_orig.weights[3][i]
    end
    return new_net
  end



function run_individual(invd_idx)
  local speed_bonus = 0
  local time_out_const = 80
  local time_out = time_out_const
  local rightmost = 0
  local framecount = 0
  local score = 0
  savestate.load(FILENAME)
  while time_out > 0 do
    print_fitness()
    --clear_joypad()
    if framecount%5==0 then
      local inputs = get_inputs()
      local outputs = evaluate_network(population.individuals[invd_idx].network,inputs)
      controller = outputs
      if controller["P1 Left"] and controller["P1 Right"] then
        controller["P1 Left"] = false
        controller["P1 Right"] = false
      end
      if controller["P1 Up"] and controller["P1 Down"] then
        controller["P1 Up"] = false
        controller["P1 Down"] = false
      end
    end
    controller["P1 Y"] = true
    joypad.set(controller)
    get_positions()
    if mario_x > rightmost then
      rightmost = mario_x
      time_out = time_out_const
    end
    score = memory.read_s16_le(0x0F34)
    time_out = time_out-1
    speed_bonus= math.floor(mario_x/framecount)
    population.individuals[invd_idx].fitness= mario_x + speed_bonus
    if rightmost > 4816 then
      print("Winner!")
      population.individuals[invd_idx].fitness = 10000
    end
    framecount=framecount+1
    emu.frameadvance()
  end
  --print("score:", score)
end

function print_update_fitness_rank()
  print("-- Fitness --")
  for i=1,POPULATION_NR do
    population.individuals[i].rank = i
    print(population.individuals[i].fitness .. ", " .. population.individuals[i].id)
    if population.individuals[i].prev_fitness == population.individuals[i].fitness then
      population.individuals[i].same_fitness_count = population.individuals[i].same_fitness_count +1
    else
      population.individuals[i].same_fitness_count =0
    end
    population.individuals[i].prev_fitness = population.individuals[i].fitness
  end
  print("-------------")
end

  function average_fitness(nr_of_p)
    local avrg = 0
    for i=1,nr_of_p do
      avrg = avrg + population.individuals[i].fitness
    end
    avrg = avrg/nr_of_p
    return avrg
  end

  function print_fitness()
    local x_axis_span = 100
    local y_axis_span = 100
    local x1 = 2
    local x2 = x1 + x_axis_span
    local y1 = 8
    local y2 = y1 + y_axis_span
    local max_x = best_so_far
    gui.drawBox(0,0,120,115,0xD0FFFFFF,0xD0FFFFFF)
    gui.drawLine(x1, y1, x1, y2,"black")
    gui.drawLine(x1, y2, x2, y2,"black")
    gui.drawText(6, 0, "best:" .. best_so_far, "blue", 0, 10)
    x_step = math.floor(x_axis_span/generation_count)
    if x_step < 1 then
      x_step=1
    end
    local cord_prev_x = x1
    local cord_prev_y = y2
    for i=1,generation_count do
      local cord_2_x = x1 + (i*x_step)
      local cord_2_y = y2 - math.floor(y_axis_span * (fitness_values[i]/best_so_far))
      gui.drawLine(cord_prev_x, cord_prev_y, cord_2_x, cord_2_y,"black")
      if generation_count < 10 then
        gui.drawEllipse(cord_2_x, cord_2_y, 2, 2, "green", "green")
      elseif i%5==0 and i<50 then
        gui.drawEllipse(cord_2_x, cord_2_y, 2, 2, "green", "green")
      elseif i%10==0 and i<100 then
        gui.drawEllipse(cord_2_x, cord_2_y, 2, 2, "green", "green")
      end
      cord_prev_x = x1 + (i*x_step)
      cord_prev_y = math.floor(y2 - (y_axis_span * (fitness_values[i]/best_so_far)))
    end
    cord_prev_x = x1
    cord_prev_y = y2
    for i=1,generation_count do
      local cord_2_x = x1 + (i*x_step)
      local cord_2_y = y2 - math.floor(y_axis_span * (best_fitness_values[i]/best_so_far))
      gui.drawLine(cord_prev_x, cord_prev_y, cord_2_x, cord_2_y,"red")
      if generation_count < 10 then
        gui.drawEllipse(cord_2_x, cord_2_y, 2, 2, "green", "green")
        gui.drawText(cord_2_x, cord_2_y, "g" .. i, "blue", 0, 10)
      elseif i%5==0 and i<50 then
        gui.drawEllipse(cord_2_x, cord_2_y, 2, 2, "green", "green")
        gui.drawText(cord_2_x, cord_2_y, "g" .. i, "blue", 0, 10)
      elseif i%10==0 and i<101 then
        gui.drawEllipse(cord_2_x, cord_2_y, 2, 2, "green", "green")
        gui.drawText(cord_2_x, cord_2_y, "g" .. i, "blue", 0, 10)
      end
      cord_prev_x = x1 + (i*x_step)
      cord_prev_y = math.floor(y2 - (y_axis_span * (best_fitness_values[i]/best_so_far)))
    end
  end
  -------------RUN----------------------
  population = new_population()
  old_average = 0
  best_so_far = 1
  controller = {}
  generation_count = 1
  fitness_values = {1}
  best_fitness_values = {1}
  while true do
    for i=1,POPULATION_NR do
      clear_joypad()
      print("running individual:", population.individuals[i].id)
      run_individual(i)
    end
    table.sort(population.individuals, function (a,b)
      return (a.fitness > b.fitness)
    end)
    print_update_fitness_rank()
    if population.individuals[1].fitness > best_so_far then
      best_so_far = population.individuals[1].fitness
      print("-- New best: " .. population.individuals[1].id .. ", fitness: " .. best_so_far)
      --print("with weight sum:", weight_sum(population.individuals[1].network))
    end
    fitness_values[#fitness_values+1] = average_fitness(NR_OF_PARENTS_TO_BREED_FROM)
    best_fitness_values[#best_fitness_values+1]= best_so_far
    --for i=1,#fitness_values do
    --  print(fitness_values[i])
    --end
    population = evolve_2(population)
    generation_count = generation_count + 1
    print("preparing generation " .. generation_count .. "...")
    --print("rank 1 weight sum:", weight_sum(population.individuals[1].network))
    sleep(2)
  end
