# README 

This repo contains an attempt at a command line app that can perform basic deep Q-learning on the open-ai platform. It is a hobby project, nothing more nothing less. 

I've added some tests, which you can simply run by calling `pytest` from the home folder.

You can run the command line via; 

```
python commandline
```

If you're wondering about settings, there's some stuff you can pass along:

```
python commandline --help
```

If you want to change the actual neural network though, you'll need to change code in the `ValueModel` object in the `StateFitter.py` file. 
