import sys

with open(sys.argv[1], "r") as file:
	s = file.readlines()
	dic = {}
	num = []
	indicator = 0
	for i in s:
		words = i.replace("\n", "").split(" ")
		for word in words:
			if word not in dic:
				dic[word] = 1
				num.append(word)
			else:
				dic[word] += 1
	output = []
	for i, word in enumerate(num):
		output.append([word, str(i), str(dic[word])])
	out = "\n".join([" ".join(i) for i in output])

	with open("Q1.txt", "w") as file2:
		file2.write(out)
