`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/17/2026 11:38:03 AM
// Design Name: 
// Module Name: tb_top
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
// 
//////////////////////////////////////////////////////////////////////////////////


`timescale 1ns/1ps
// ─────────────────────────────────────────────────────────────
// tb_top.sv
// Testbench for top_level.sv
// Tests digit 8 classification
// ─────────────────────────────────────────────────────────────
module tb_top;

    // ── parameters ────────────────────────────────────────────
    localparam INPUT_SIZE    = 784;
    localparam HIDDEN_SIZE   = 128;
    localparam OUTPUT_SIZE   = 10;
    localparam NUM_STEPS     = 25;
    localparam BITWIDTH      = 16;
    localparam THRESHOLD     = 1;      // matches your lif.sv threshold
    localparam CLK_PERIOD    = 10;     // 10ns = 100 MHz
    localparam EXPECTED_DIGIT = 8;     // testing digit 8

    // ── signals ───────────────────────────────────────────────
    reg                            clk;
    reg                            rst;
    reg                            start;
    reg  [INPUT_SIZE-1:0]          spike_in;
    wire [3:0]                     predicted_class; // $clog2(10) = 4 bits
    wire                           done;

    // ── DUT ───────────────────────────────────────────────────
    top_level #(
        .INPUT_SIZE  (INPUT_SIZE),
        .HIDDEN_SIZE (HIDDEN_SIZE),
        .OUTPUT_SIZE (OUTPUT_SIZE),
        .BITWIDTH    (BITWIDTH),
        .THRESHOLD   (THRESHOLD),
        .NUM_STEPS   (NUM_STEPS)
    ) dut (
        .clk             (clk),
        .rst             (rst),
        .start           (start),
        .spike_in        (spike_in),
        .predicted_class (predicted_class),
        .done            (done)
    );

    // ── clock ─────────────────────────────────────────────────
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ── load spike pattern from .mem file ────────────────────
    // test_spike_t0.mem: 49 lines of 16-bit hex
    // 49 × 16 bits = 784 bits = one spike frame for digit 8
    reg [15:0] spike_words  [0:48];
    reg [INPUT_SIZE-1:0] spike_pattern;
    integer k;

    initial begin
        $readmemh("test_spike_t0.mem", spike_words);
        spike_pattern = 0;
        for (k = 0; k < 49; k = k + 1)
            spike_pattern[k*16 +: 16] = spike_words[k];
        $display("[TB] test_spike_t0.mem loaded successfully");
    end

    // ── waveform dump ─────────────────────────────────────────
    initial begin
        $dumpfile("tb_top.vcd");
        $dumpvars(0, tb_top);
    end

    // ── task: run one inference and check result ──────────────
    // using clock-edge based done detection - no infinite loop risk
    task run_inference;
        input integer expected;
        input integer inference_num;
        integer cycle_count;
        integer found_done;
        begin
            // pulse start for one cycle
            @(posedge clk); #1;
            start = 1;
            @(posedge clk); #1;
            start = 0;

            // wait for done - check at every posedge
            // FSM needs: NUM_STEPS + 3 extra cycles (CLASSIFY + DONE_ST)
            // safe timeout = NUM_STEPS + 10 = 35 cycles
            found_done  = 0;
            cycle_count = 0;

            repeat(NUM_STEPS + 10) begin
                @(posedge clk); #1;
                cycle_count = cycle_count + 1;
                if (done) begin
                    found_done = 1;
                end
            end

            // print result
            $display("─────────────────────────────────────────────");
            $display("[TB] Inference %0d complete after %0d cycles",
                      inference_num, cycle_count);
            if (found_done) begin
                $display("[TB] done signal    : RECEIVED ✅");
                $display("[TB] predicted_class: %0d", predicted_class);
                $display("[TB] expected_class : %0d", expected);
                if (predicted_class == expected)
                    $display("[TB] RESULT         : CORRECT ✅");
                else
                    $display("[TB] RESULT         : WRONG ❌");
            end
            else begin
                $display("[TB] done signal    : NOT RECEIVED ❌");
                $display("[TB] predicted_class: %0d (may be invalid)",
                          predicted_class);
                $display("[TB] Check FSM in top_level.sv");
            end
            $display("─────────────────────────────────────────────");
        end
    endtask

    // ── main stimulus ─────────────────────────────────────────
    initial begin
        // initialise all signals
        rst      = 1;
        start    = 0;
        spike_in = 0;

        $display("=============================================");
        $display("[TB] SNN Testbench - Digit %0d classification",
                  EXPECTED_DIGIT);
        $display("[TB] Architecture: %0d → %0d → %0d",
                  INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        $display("[TB] Timesteps   : %0d", NUM_STEPS);
        $display("[TB] Threshold   : %0d", THRESHOLD);
        $display("=============================================");

        // hold reset for 5 cycles
        repeat(5) @(posedge clk);
        #1;
        rst = 0;
        $display("[TB] Reset released");

        // wait 2 cycles after reset before driving input
        repeat(2) @(posedge clk);
        #1;

        // set spike_in - held constant for all 25 timesteps
        // paper Section 3.2: same image repeated each timestep
        // different spike patterns per pixel come from Bernoulli
        // probability already baked into test_spike_t0.mem
        spike_in = spike_pattern;
        $display("[TB] spike_in loaded - running inference 1...");

        // ── inference 1 ───────────────────────────────────────
        run_inference(EXPECTED_DIGIT, 1);

        // wait 3 cycles between inferences
        repeat(3) @(posedge clk);
        #1;

        // ── inference 2 - verify reset works ─────────────────
        $display("[TB] Running inference 2 to verify system reset...");
        run_inference(EXPECTED_DIGIT, 2);

        // done
        repeat(5) @(posedge clk);
        $display("[TB] Simulation complete");
        $display("=============================================");
        $finish;
    end

endmodule
