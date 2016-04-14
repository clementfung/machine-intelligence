import subprocess
import time

t1 = "apple"
t2 = "orchard"

init_command = "./run_analyzer"
fw = open("tmpout", "wb")
fr = open("tmpout", "r")

p = subprocess.Popen(init_command, stdin=subprocess.PIPE, stdout=fw, stderr=fw)
time.sleep(5)

p.stdin.write("apple\norchard\n")
time.sleep(1)

out = fr.read()

print out

fw.close()
fr.close()