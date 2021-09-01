
with open("dayeight.txt") as file:
	loop = file.read().split("\n")

# Defines method to run through instructions, running until an instruction is repeated
# or the end of the instruction set is reached, at which point the method results in
# an index out of bounds error used to determine the correct fix for the problem
def instr(line, acc, loop):
	try:
		loop[line]
	except:
		return True, acc

	order = loop[line]
	order = order.split(" ")
	ogline = line
	if order[0].startswith("*"):
		# print(acc) "Used to return answer for part 1"
		return False, acc
	if order[0] == "acc":
		if order[1].startswith("+"):
			acc = acc + int(order[1][1:])
		else:
			acc = acc - int(order[1][1:])

	elif order[0] == "jmp":
		if order[1].startswith("+"):
			line = line + int(order[1][1:]) - 1
		else:
			line = line - int(order[1][1:]) - 1


	linelist.append(ogline)
	loop[ogline] = "*" + loop[ogline]
	
	return instr(line + 1, acc, loop)


# Defines method used to find the correct fix for the infinite loop. Replaces each
# instruction in the initial loop in a brute force manner since 
# the problematic instruction must be in the inital loop
def findLoop(loop, line):
	loop = original.copy()
	linelist = firsttrylines.copy()
	if loop[line].split()[0] == "jmp":
		loop[line] = "nop " + loop[line].split()[1]
	elif loop[line].split()[0] == "nop":
		loop[line] = "jmp " + loop[line].split()[1]

	done, acc = instr(0, 0, loop)
	if done:
		print("Final acc = {}".format(acc))
		return True
	return False



original = loop.copy()
linelist = []
instr(0, 0, loop)
firsttrylines = linelist.copy()
for x in range(len(linelist)):
	if findLoop(loop, linelist[x]):
		break

