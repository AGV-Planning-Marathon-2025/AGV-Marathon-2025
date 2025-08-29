If you face an error saying "Could Not Display AutoDRIVE Simulator.x86_64" on Ubuntu:

Set the executable flag on the process for `AutoDRIVE Simulator.x86_64` by running `chmod` on the executable file (won't hurt if it's already set)

```bash
$ cd <path/to/AutoDRIVE Simulator.x86_64>
$ sudo chmod +x AutoDRIVE\ Simulator.x86_64
```
