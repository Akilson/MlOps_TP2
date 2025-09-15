#!/bin/sh

echo "test with body: {size: 100, rooms:2, is_garden:0}\n"
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"size": 100, "rooms": 2, "is_garden": 0}'
echo "\n"

echo "test with body: {size: 150, rooms:3, is_garden:1}\n"
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"size": 150, "rooms": 3, "is_garden": 1}'
echo "\n"
