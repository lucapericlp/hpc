#include <stdio.h>
#include <stdint.h>

/**
 * High-Performance Computing Pattern: Bit-Packing via Bitfields
 *
 * In RL environments and multi-agent simulations, memory bandwidth and cache
 * pressure are often bottlenecks. Packing multiple flags or small integers
 * into a single byte reduces the data transfer size between the simulation
 * and the neural network.
 */

// A packed structure representing agent state
typedef struct {
    uint8_t health : 4;    // 0-15 (4 bits)
    uint8_t team   : 2;    // 0-3  (2 bits)
    uint8_t alive  : 1;    // 0-1  (1 bit)
    uint8_t active : 1;    // 0-1  (1 bit)
} AgentState;

int main() {
    // Total size of the struct should be 1 byte (8 bits)
    printf("Size of AgentState: %lu byte\n", sizeof(AgentState));

    AgentState player;
    player.health = 10;    // 1010 in binary
    player.team = 3;      // 11 in binary
    player.alive = 1;     // 1 in binary
    player.active = 0;    // 0 in binary


    printf("Player Health: %u\n", player.health);
    printf("Player Team:   %u\n", player.team);
    printf("Is Alive:      %s\n", player.alive ? "Yes" : "No");
    printf("Is Active:     %s\n", player.active ? "Yes" : "No");

    // Inspect the raw byte representation
    // &player is a pointer to the struct, we cast it to a uint8_t pointer and dereference it
    // to get the packed byte value.
    uint8_t raw = *(uint8_t*)&player;
    printf("\nExpected: 0x7A, Got raw byte (Hex): 0x%02X\n", raw);

    // Expected layout in memory (LSB to MSB depends on endianness/compiler):
    // active(1) alive(1) team(2) health(4)
    // 0         1        11      1010  => 01111010 in binary (0x7A)

    return 0;
}
