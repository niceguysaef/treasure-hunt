import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np

# --- World Setup ---
world = [
    ['.', '.', '.', '.', 'R1', '.', '.', '.', '.', '.'],
    ['.', 'T2', '.', 'T4', '$', '.', 'T3', '.', '#', '.'],
    ['.', '.', '#', '.', '#', '.', '.', 'R2', 'T1', '.'],
    ['#', 'R1', '.', '#', '.', 'T3', '#', '$', '.', '$'],
    ['.', '.', 'T2', '$', '#', '.', '#', '#', '.', '.'],
    ['.', '.', '.', '.', '.', 'R2', '.', '.', '.', '.']
]

ROWS = len(world)
COLS = len(world[0])

strict_offset_col_directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
strict_alt_col_directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]

frozen_states = set()

def in_bounds(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS

def get_treasures():
    return {(r, c) for r in range(ROWS) for c in range(COLS) if world[r][c] == '$'}

def offset_to_cube(row, col):
    x = col - (row - (row % 2)) // 2
    z = row
    y = -x - z
    return (x, y, z)

def hex_distance(a, b):
    ax, ay, az = offset_to_cube(*a)
    bx, by, bz = offset_to_cube(*b)
    return max(abs(ax - bx), abs(ay - by), abs(az - bz))

def mst_heuristic(current, treasures):
    if not treasures:
        return 0
    unvisited = set(treasures)
    visited = {current}
    total_cost = 0
    while unvisited:
        min_edge = float('inf')
        next_node = None
        for v in visited:
            for u in unvisited:
                d = hex_distance(v, u)
                if d < min_edge:
                    min_edge = d
                    next_node = u
        visited.add(next_node)
        unvisited.remove(next_node)
        total_cost += min_edge
    return total_cost

def is_valid_move(x, y, dx, dy):
    nx, ny = x + dx, y + dy
    return in_bounds(nx, ny) and world[nx][ny] != '#'

def apply_tile_effect(tile, speed, energy, direction, x, y, collected, original_treasures, step_multiplier, step_decay, prev_tile):
    new_positions = [(x, y)]
    new_collected = collected.copy()
    new_speed = speed
    new_energy = energy
    new_multiplier = step_multiplier
    new_step_decay = step_decay
    reward1_triggered = False
    triggered_t4 = False 

    if tile == 'T1':
        new_multiplier = 1
    elif tile == 'T2':
        if prev_tile != 'T2' and (x, y) not in frozen_states:
            frozen_states.add((x, y))
            new_positions = [(x, y), (x, y)]
            new_speed *= 0.5
    elif tile == 'T3':
        dx, dy = direction
        current_x, current_y = x, y
        for _ in range(2):
            next_x, next_y = current_x + dx, current_y + dy
            if not in_bounds(next_x, next_y) or world[next_x][next_y] == '#':
                return None
            new_positions.append((next_x, next_y))
            current_x, current_y = next_x, next_y
            if world[current_x][current_y] == '$' and (current_x, current_y) in original_treasures:
                new_collected.add((current_x, current_y))
            new_energy -= 1
            if new_energy <= 0:
                return None
        x, y = current_x, current_y
    elif tile == 'T4':
        triggered_t4 = True
    elif tile == 'R1':
        reward1_triggered = True
    elif tile == 'R2':
        if prev_tile != 'R2':
            new_speed *= 2

    if world[x][y] == '$' and (x, y) in original_treasures:
        new_collected.add((x, y))

    return new_positions, new_speed, new_energy, new_collected, new_multiplier, new_step_decay, reward1_triggered, triggered_t4

def a_star():
    original_treasures = get_treasures()
    start = (0, 0)
    initial_state = (0, start[0], start[1], 1.0, 100.0, frozenset(), (0, 1), 1, 1.0, False, frozenset(original_treasures), 1.0, -1)
    visited = {}
    came_from = {}
    visit_count = {}
    heap = [(0, initial_state)]
    move_counter = 0
    best_path = None
    best_cost = float('inf')
    best_state = None

    while heap:
        if move_counter > 10000:
            print("Aborted: too many moves (>10000)")
            break

        f, (cost, x, y, speed, energy, collected, direction, step_multiplier, step_decay, reward_mode, available_treasures, r1_step_energy, t1_step_count) = heapq.heappop(heap)
        state_id = (x, y, collected, available_treasures, round(step_multiplier, 3), round(step_decay, 3), reward_mode, round(r1_step_energy, 3), t1_step_count)
        if state_id in visited and visited[state_id] <= cost:
            continue
        visited[state_id] = cost

        if collected == original_treasures and cost < best_cost:
            best_path, best_energy_log = reconstruct_path(came_from, (
                cost, x, y, speed, energy, collected, direction, step_multiplier,
                step_decay, reward_mode, available_treasures, r1_step_energy, t1_step_count))
            best_cost = cost
            best_state = (cost, x, y, speed, energy, collected, direction,
                          step_multiplier, step_decay, reward_mode, available_treasures, r1_step_energy, t1_step_count)
        directions = strict_offset_col_directions if y % 2 == 0 else strict_alt_col_directions

        for dx, dy in directions:
            if not is_valid_move(x, y, dx, dy):
                continue
            nx, ny = x + dx, y + dy
            tile = world[nx][ny]
            result = apply_tile_effect(tile, speed, energy, (dx, dy), nx, ny, set(collected),
                                       original_treasures, step_multiplier, step_decay, world[x][y])
            if result is None:
                continue
            new_positions, new_speed, new_energy, new_collected, new_multiplier, new_decay, triggered_r1, triggered_t4 = result
            new_available_treasures = available_treasures
            if triggered_t4:
                new_available_treasures = frozenset(new_collected)
            next_reward_mode = reward_mode or triggered_r1
            next_r1_step_energy = r1_step_energy

            # Track trap 1 activation
            if tile == 'T1' and t1_step_count == -1:
                next_t1_step_count = 0
            elif t1_step_count >= 0:
                next_t1_step_count = t1_step_count + 1
            else:
                next_t1_step_count = -1

            move_cost = 0
            energy_cost = 0
            for i, pos in enumerate(new_positions):
                tile_type = world[pos[0]][pos[1]]
                trap_penalty = {'T1': 5, 'T2': 10, 'T3': 20, 'T4': 25}.get(tile_type, 0)
                revisit_penalty = 5 * visit_count.get(pos, 0)

                # Energy cost logic
                if next_reward_mode:
                    step_energy = max(0.125, next_r1_step_energy)
                    next_r1_step_energy = max(0.125, next_r1_step_energy / 2)
                elif next_t1_step_count >= 0:
                    step_energy = min(2 ** next_t1_step_count, 4)
                else:
                    step_energy = new_multiplier * (2 ** i) * new_decay

                move_cost += 1 + trap_penalty + revisit_penalty
                energy_cost += step_energy
                visit_count[pos] = visit_count.get(pos, 0) + 1

            new_energy -= energy_cost
            if new_energy <= 0:
                continue

            new_cost = cost + move_cost
            h = mst_heuristic((new_positions[-1][0], new_positions[-1][1]),
                              list(set(new_available_treasures) - new_collected))
            total_f = new_cost + h

            prev_state = (cost, x, y, speed, energy, collected, direction, step_multiplier,
                          step_decay, reward_mode, available_treasures, r1_step_energy, t1_step_count)
            for pos in new_positions[:-1]:
                intermediate_state = (cost, pos[0], pos[1], speed, energy, frozenset(new_collected),
                                      direction, new_multiplier, new_decay, next_reward_mode,
                                      new_available_treasures, next_r1_step_energy, next_t1_step_count)
                came_from[intermediate_state] = prev_state
                prev_state = intermediate_state

            new_state = (new_cost, new_positions[-1][0], new_positions[-1][1], new_speed, new_energy,
                         frozenset(new_collected), (dx, dy), new_multiplier, new_decay,
                         next_reward_mode, new_available_treasures, next_r1_step_energy, next_t1_step_count)
            came_from[new_state] = prev_state
            heapq.heappush(heap, (total_f, new_state))
            move_counter += 1



    if best_path:
        step_counter = 0.0
        in_reward_2 = False
        print("Path taken:")
        for i, step in enumerate(best_path):
            r, c = step
            tile = world[r][c]
            action = {
                '$': "Collected treasure",
                'T1': "Entered Trap 1 (start exponential energy cost)",
                'T2': "Entered Trap 2 (half speed + frozen step)",
                'T3': "Entered Trap 3 (forced movement)",
                'T4': "Entered Trap 4 (reset treasures)",
                'R1': "Entered Reward 1 (begin halving energy cost per step)",
                'R2': "Entered Reward 2 (double speed)",
                '#': "Hit obstacle",
                '.': "Moved to empty tile"
            }.get(tile, "Unknown action")
            tile_label = world[r][c]
            energy_now = best_energy_log[i]
            if tile_label == 'R2':
                print(f"Step {step_counter:.1f} at Reward 2: {step} - {action} | Energy left: {energy_now:.4f}")
                in_reward_2 = True
            elif tile_label == 'T2':
                print(f"Step {step_counter:.1f} at Trap 2: {step} - {action} | Energy left: {energy_now:.4f}")
                step_counter += 2.0
                in_reward_2 = False
                continue
            else:
                print(f"Step {step_counter:.1f}: {step} - {action} | Energy left: {energy_now:.4f}")
            if in_reward_2:
                step_counter += 0.5
                in_reward_2 = False
            else:
                step_counter += 1.0
        print("Total moves:", move_counter)
        print(f"Final tile coordinate: {best_path[-1]} - Tile type: {world[best_path[-1][0]][best_path[-1][1]]}")
        print(f"Final energy: {best_state[4]:.4f}, Final speed: {best_state[3]:.4f}")
        print(f"All treasures collected with total cost: {best_cost}")
        plot_path(best_path)
    else:
        print("Failed to collect all treasures.")

def reconstruct_path(came_from, end_state):
    path = []
    energy_log = []
    current = end_state
    while current in came_from:
        path.append((current[1], current[2]))
        energy_log.append(current[4])  # Capture energy at this step
        current = came_from[current]
    path.append((0, 0))
    energy_log.append(current[4])
    path.reverse()
    energy_log.reverse()
    return path, energy_log

def plot_path(path):
    fig, ax = plt.subplots(figsize=(12, 8))
    size = 1
    coord_counts = {}
    step_counter = 0.0
    in_reward_2 = False

    for r in range(ROWS):
        for c in range(COLS):
            x = c * 3/2
            y = np.sqrt(3) * ((ROWS - 1 - r) + 0.5 * (c % 2))
            tile = world[r][c]
            color = 'white'
            if tile == '#':
                color = 'black'
            elif tile == '$':
                color = 'gold'
            elif tile.startswith('T'):
                color = 'salmon'
            elif tile.startswith('R'):
                color = 'lightblue'
            hex = RegularPolygon((x, y), numVertices=6, radius=size / np.sqrt(3), orientation=np.radians(30), facecolor=color, edgecolor='black')
            ax.add_patch(hex)
            ax.text(x, y, tile, ha='center', va='center', fontsize=8)

    for (r, c) in path:
        x = c * 3/2
        base_y = np.sqrt(3) * ((ROWS - 1 - r) + 0.5 * (c % 2))
        tile_label = world[r][c]

        if (r, c) not in coord_counts:
            coord_counts[(r, c)] = 0
        else:
            coord_counts[(r, c)] += 1
        offset = coord_counts[(r, c)] * -0.25

        ax.text(x, base_y + offset, f"{step_counter:.1f}", ha='center', va='center', fontsize=8, color='blue', weight='bold')

        if tile_label == 'R2':
            in_reward_2 = True
        elif tile_label == 'T2':
            step_counter += 2.0
            in_reward_2 = False
            continue

        if in_reward_2:
            step_counter += 0.5
            in_reward_2 = False
        else:
            step_counter += 1.0

    ax.set_aspect('equal')
    ax.autoscale_view()
    plt.axis('off')
    plt.title("Optimized Treasure Hunt Path")
    plt.show()

a_star()
