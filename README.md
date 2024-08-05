# Warehouse Automation with Smart Robots

Designing an optimized path for multiple robots in a warehouse for picking and delivery operations using OCR and path finding algorithms: A\* algorithm (shortest path) and Djikstra algorithm.

### There are 3 scenarios

- One Robot delivering to multiple locations with maximum capacity
- One Robot delivering to multiple locations by calculating shortest travel path
- Multiple robots to multiple locations

### Warehouse Environment

- Initial Setup: All robots start at the starting station (GREY). Set the location for one robot (PINK).
- Stocking Shelves: Represented by BLACK.
- Robots: Each robot is indicated by a different color.
- Delivery Locations: Marked in GREEN.

### Total Minimum Cost

- Shortest path distance is calculated using A\* algorithm for every robot-task pair
- Minimum objective = sum_i,sum_j(r_i\*cost(deliver_j))
- Constraints: deliver_j can be allocated to only unique robot

### Comparison between A\* and Dijkstra Algorithm for finding the shortest path between robot and an end point

- Efficiency: A\* is faster because it avoids searching unnecessary nodes.
- Informed Search: A\* is an informed search algorithm, taking the goal state into account, unlike Dijkstra.
