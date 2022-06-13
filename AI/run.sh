
#!/bin/bash
echo "training"
nohup python main.py > ~/O2E/O2E-TU-2/running.log  2>&1 & exit
