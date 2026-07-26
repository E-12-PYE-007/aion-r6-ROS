Build and run:
```
cd eval/controller_eval
cmake -B build -S . -DCMAKE_PREFIX_PATH=/opt/ros/humble
cmake --build build
python3 test_basic_chunk.py
```