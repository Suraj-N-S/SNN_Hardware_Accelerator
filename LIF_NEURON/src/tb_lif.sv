`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/16/2026 12:22:21 PM
// Design Name: 
// Module Name: tb_lif
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


module tb_lif;

    // parameters 
    localparam BITWIDTH  = 16;
    localparam THRESHOLD = 100;
    localparam N_INPUTS  = 4;

    reg                        clk;
    reg                        rst;
    reg  [N_INPUTS-1:0]        spike_in;
    wire                       spike_out;

    // weights 
    wire signed [BITWIDTH-1:0] weights [0:N_INPUTS-1];
    assign weights[0] = 30;
    assign weights[1] = 25;
    assign weights[2] = 40;
    assign weights[3] = 20;

    // instantiate the neuron
    lif #(
        .BITWIDTH (BITWIDTH),
        .THRESHOLD(THRESHOLD),
        .N_INPUTS (N_INPUTS)
    ) uut (
        .clk      (clk),
        .rst      (rst),
        .spike_in (spike_in),
        .weights  (weights),
        .spike_out (spike_out)
    );  
    
    //clock : period of 10ns
    always #5 clk = ~clk; 
    
    integer cycle;
    initial cycle = 0;
    
    always @(posedge clk) begin 
    
        if(!rst) begin 
            #1; //small delay 
            $display("cycle %0d | spike_in= %b | fired = %b",cycle,spike_in,spike_out);
            cycle = cycle + 1;
        end 
        
    end 
    
    initial begin 
    
        clk = 0;
        rst = 1;
        spike_in = 4'b0000;
        
        #20 rst = 0; 
        
        #10 spike_in = 4'b0101;
        #10 spike_in = 4'b0101;
        #10 spike_in = 4'b0000; 
        
        #30
        
        spike_in = 4'b1111;
        #10 spike_in = 4'b0000;
        
        #50 
        
        $display("simulation completed");
        $finish;
        
     end
    
    

    
endmodule
