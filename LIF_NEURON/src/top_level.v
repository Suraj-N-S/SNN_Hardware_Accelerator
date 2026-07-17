`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/17/2026 11:15:44 AM
// Design Name: 
// Module Name: top_level
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// ─────────────────────────────────────────────────────────────
// top_level.sv
// Connects two SNN layers and a spike counter
// Follows arXiv:2411.01628 Figure 5 and Figure 6
//
// Architecture:
//   784-bit spike_in
//       ↓ snn_layer1 (128 LIF neurons, weights_layer1.mem)
//   128-bit hidden_spikes
//       ↓ snn_layer2 (10 LIF neurons, weights_layer2.mem)
//   10-bit output_spikes
//       ↓ spike counter over 25 timesteps
//   predicted_class (0-9)
//
// Paper Figure 6: three phases of hardware execution
//   Phase 1 : hidden layer computation
//   Phase 2 : output layer computation
//   Phase 3 : comparator finds winning class
// ─────────────────────────────────────────────────────────────
//////////////////////////////////////////////////////////////////////////////////


module top_level #(
    parameter INPUT_SIZE   = 784,    // one spike per pixel
    parameter HIDDEN_SIZE  = 128,    // hidden LIF neurons
    parameter OUTPUT_SIZE  = 10,     // output neurons - one per digit
    parameter BITWIDTH     = 16,     // Q1.15 weight format
    parameter THRESHOLD    = 1,  
    parameter NUM_STEPS    = 25      // timesteps - paper Section 4.2.1
)(
    input  wire                              clk,
    input  wire                              rst,
    input  wire                              start,         // pulse high to begin inference
    input  wire [INPUT_SIZE-1:0]             spike_in,      // 784-bit rate coded input
    output reg  [$clog2(OUTPUT_SIZE)-1:0]    predicted_class, // 0-9
    output reg                               done           // high for 1 cycle when result ready
);
 
    // ── internal signals ──────────────────────────────────────
    wire [HIDDEN_SIZE-1:0]  hidden_spikes;   // layer1 → layer2
    wire [OUTPUT_SIZE-1:0]  output_spikes;   // layer2 → counter
 
    // ── FSM states ────────────────────────────────────────────
    // paper Figure 6: Phase 1 (hidden), Phase 2 (output), Phase 3 (compare)
    localparam IDLE     = 2'd0;
    localparam RUNNING  = 2'd1;
    localparam CLASSIFY = 2'd2;
    localparam DONE_ST  = 2'd3;
 
    reg [1:0]                        state;
    reg [$clog2(NUM_STEPS)-1:0]      step_count;
    reg                              running;
 
    // spike counters - one per output neuron
    // 5 bits wide - enough for max 25 spikes
    reg [4:0] spike_count [0:OUTPUT_SIZE-1];
 
    integer n;
 
    // ── SNN layer 1 - hidden layer ────────────────────────────
    // paper Figure 4: input layer (784) → hidden LIF (128)
    // paper Section 4.3: cascaded adder feeds LIF neurons
    snn_layer #(
        .NUM_NEURONS (HIDDEN_SIZE),
        .NUM_INPUTS  (INPUT_SIZE),
        .BITWIDTH    (BITWIDTH),
        .THRESHOLD   (THRESHOLD),
        .MEM_FILE    ("W1.mem")
    ) layer1 (
        .clk      (clk),
        .rst      (rst | ~running),  // reset neurons when not running
        .spike_in (spike_in),
        .spike_out(hidden_spikes)
    );
 
    // ── SNN layer 2 - output layer ────────────────────────────
    // paper Figure 4: hidden LIF (128) → output LIF (10)
    snn_layer #(
        .NUM_NEURONS (OUTPUT_SIZE),
        .NUM_INPUTS  (HIDDEN_SIZE),
        .BITWIDTH    (BITWIDTH),
        .THRESHOLD   (THRESHOLD),
        .MEM_FILE    ("W2.mem")
    ) layer2 (
        .clk      (clk),
        .rst      (rst | ~running),  // reset neurons when not running
        .spike_in (hidden_spikes),
        .spike_out(output_spikes)
    );
 
    // ── FSM ───────────────────────────────────────────────────
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state           <= IDLE;
            step_count      <= 0;
            running         <= 0;
            done            <= 0;
            predicted_class <= 0;
            for (n = 0; n < OUTPUT_SIZE; n = n + 1)
                spike_count[n] <= 0;
        end
        else begin
            case (state)
 
                // ── wait for start ────────────────────────────
                IDLE: begin
                    done    <= 0;
                    running <= 0;
                    if (start) begin
                        // clear spike counters for fresh inference
                        for (n = 0; n < OUTPUT_SIZE; n = n + 1)
                            spike_count[n] <= 0;
                        step_count <= 0;
                        running    <= 1;
                        state      <= RUNNING;
                    end
                end
 
                // ── run 25 timesteps - paper Section 4.2.1 ───
                // paper Figure 6 Phase 1 and Phase 2
                RUNNING: begin
                    // count output spikes each cycle
                    // paper Section 4.3: output memory as shift register
                    for (n = 0; n < OUTPUT_SIZE; n = n + 1) begin
                        if (output_spikes[n])
                            spike_count[n] <= spike_count[n] + 1;
                    end
 
                    if (step_count == NUM_STEPS - 1) begin
                        running    <= 0;
                        state      <= CLASSIFY;
                    end
                    else begin
                        step_count <= step_count + 1;
                    end
                end
 
                // ── find neuron with most spikes ─────────────
                // paper Figure 6 Phase 3: comparator
                // paper Figure 5: Output Memory → Comparator → Collision/No-Collision
                CLASSIFY: begin
                    predicted_class <= 0;
                    for (n = 1; n < OUTPUT_SIZE; n = n + 1) begin
                        if (spike_count[n] > spike_count[predicted_class])
                            predicted_class <= n[$clog2(OUTPUT_SIZE)-1:0];
                    end
                    state <= DONE_ST;
                end
 
                // ── assert done for one cycle ─────────────────
                DONE_ST: begin
                    done  <= 1;
                    state <= IDLE;
                end
 
            endcase
        end
    end
 
endmodule
