`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/08/2026 10:29:14 AM
// Design Name: 
// Module Name: weight_bram
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


module weight_bram #(

    parameter NUM_NEURONS = 128,
    parameter NUM_INPUTS = 784,
    parameter MEM_FILE = "W1.mem",
    parameter BITWIDTH = 16
    
)(

    input wire  clk,
    input wire [$clog2(NUM_NEURONS)-1:0] neuron_idx,
    input wire [$clog2(NUM_INPUTS)-1:0] input_idx,
    output reg signed [BITWIDTH-1:0] weight_out
        
);

    // MEMORY ARRAY 
    // elements stored in the memory file are stored in a line(flat array)
    // for neuron index n and input index i : mem[n*NUM_INPUTS+i] 
    
    reg signed [BITWIDTH-1:0] mem [0:NUM_NEURONS*NUM_INPUTS-1]; 
    
    // load .mem file at start of simulation 
    initial begin 
    
        $readmemh(MEM_FILE,mem); 
        
        $display("[BRAM] Loaded is %s (%0d weights)",MEM_FILE,NUM_NEURONS*NUM_INPUTS);
        
    end 
    
    
    // synchronous read 
    
    always @(posedge clk) begin 
    
        weight_out <= mem[neuron_idx*NUM_INPUTS + input_idx];
        
    end 
    
    


endmodule
