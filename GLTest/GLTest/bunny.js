"use strict";

var canvas;
var gl;

var numVerticesInAllBunnyFaces;

var bunny_indices;
var bunny_vertices;
var bunny_vertex_colors;
var m;

// These six variables all have to do with passing to the vertex shader.
// Not all of them are globally required, but for consistency, I made them global.
var rotateMatrix;
var scaleMatrix;
var translateMatrix;

var rotateLoc;
var scaleLoc;
var translateLoc;

// Scale bar's value
var scaleValue;

// Axes arrays due to complications with passing in 0, 1, and 2. zAxis is unused.
var xAxis;
var yAxis;
var zAxis;

// Variables used explicitly for rotation animation
var rotateDirection;
var thetaX;
var thetaY;

// Variables used explicitly for translation animation.
// "trans" variables represent where I want the bunny to go
// "current" variables represent where the bunny is
var trans_X;
var trans_Y;
var current_X;
var current_Y;

// You probably don't want to change this function.
function loaded(data, _callback)
{
	m = loadOBJFromBuffer(data);
	console.log(m);
	bunny_indices = m.i_verts;
	bunny_vertices = m.c_verts;
	numVerticesInAllBunnyFaces = bunny_indices.length;
	bunny_vertex_colors = assign_vertex_colors(bunny_vertices);
	_callback();
}	

// Uses the same rainbow scheme as shown in the documenation.
function assign_vertex_colors(input_vertices)
{   
    var colors = [];

    // Determine the maximum and minimum positions of the vertices
    var maxes = [input_vertices[0], input_vertices[1], input_vertices[2]];
    var mins  = [input_vertices[0], input_vertices[1], input_vertices[2]];
    for(let i = 0; i < input_vertices.length; i++) {
        var third_index = i % 3;
        maxes[third_index] = input_vertices[i] > maxes[third_index] ? input_vertices[i] : maxes[third_index];
        mins[third_index]  = input_vertices[i] < mins[third_index]  ? input_vertices[i] : mins[third_index];
    }

    // Assigns the colors according to the maxes and mins and pushes them to the list
    for(let i = 0; i < input_vertices.length; i+=3) {
        var colorR = (input_vertices[i]   - mins[0])/(maxes[0] - mins[0]);
        var colorB = (input_vertices[i+1] - mins[1])/(maxes[1] - mins[1]);
        var colorG = (input_vertices[i+2] - mins[2])/(maxes[2] - mins[2]);
        colors.push(vec3(colorR, colorB, colorG));
    }
	return colors;
}

// You probably don't want to change this function.
window.onload = function init()
{
    canvas = document.getElementById( "gl-canvas" );

    gl = WebGLUtils.setupWebGL( canvas );
    if ( !gl ) { alert( "WebGL isn't available" ); }

    gl.viewport( 0, 0, canvas.width, canvas.height );
    gl.clearColor( 1.0, 1.0, 1.0, 1.0 );

	// Load OBJ file using objLoader.js functions
	// These callbacks ensure the data is loaded before rendering occurs.
	loadOBJFromPath("bunny.obj", loaded, setup_after_data_load);
}

// Keyboard event listener
window.addEventListener("keydown", function(){
	switch(event.keyCode) {
		case 38:  // up arrow key
			rotateDirection = 1;
			break;
		case 40:  // down arrow key
			rotateDirection = 2;
			break;
		case 37: // left arrow key
			rotateDirection = 3;
			break;
		case 39: // right arrow key
			rotateDirection = 4;
			break;
		case 32: // spacebar
			rotateDirection = 0;
			break;
		}
}, true);

// All primary shader initialization takes place here.
function setup_after_data_load(){

	gl.enable(gl.DEPTH_TEST);
	
    // Load shaders and initialize attribute buffers
    var program = initShaders( gl, "vertex-shader", "fragment-shader" );
    gl.useProgram( program );

    // Array element buffer
    var iBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, iBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(bunny_indices), gl.STATIC_DRAW);


    // Vertex array attribute buffer
    var vBuffer = gl.createBuffer();
    gl.bindBuffer( gl.ARRAY_BUFFER, vBuffer );
    gl.bufferData( gl.ARRAY_BUFFER, new Float32Array(bunny_vertices), gl.STATIC_DRAW );

    var vPosition = gl.getAttribLocation( program, "vPosition" );
    gl.vertexAttribPointer( vPosition, 3, gl.FLOAT, false, 0, 0 );
    gl.enableVertexAttribArray( vPosition );

    // Color array attribute buffer
    var cBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, cBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, flatten(bunny_vertex_colors), gl.STATIC_DRAW);

    var vColor = gl.getAttribLocation(program, "vColor");
    gl.vertexAttribPointer(vColor, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(vColor);

    // Transformation initializations
    rotateMatrix = mat4();
    rotateLoc = gl.getUniformLocation(program, "rotateMatrix");

    scaleMatrix = mat4();
    scaleLoc = gl.getUniformLocation(program, "scaleMatrix");

    translateMatrix = mat4();
    translateLoc = gl.getUniformLocation(program, "translateMatrix");

    // Parameter initializations
    rotateDirection = 0;
    thetaX = 0;
    thetaY = 0;

    trans_X = 0;
    trans_Y = 0;
    current_X = 0;
    current_Y = 0;

    xAxis = [1, 0, 0];
    yAxis = [0, 1, 0];
    zAxis = [0, 0, 1];

    // Event listeners for buttons
    document.getElementById( "frontButton" ).onclick = function () {
        thetaX = 0;
        thetaY = 0;
        rotateDirection = 0;
        rotateMatrix = mat4();
        gl.uniformMatrix4fv(rotateLoc, false, flatten(rotateMatrix));
    };

    document.getElementById( "backButton" ).onclick = function () {
        thetaX = 0;
        thetaY = 180;
        rotateDirection = 0;
		rotateMatrix = rotate(thetaY, yAxis);
        gl.uniformMatrix4fv(rotateLoc, false, flatten(rotateMatrix));
    };

    document.getElementById( "topButton" ).onclick = function () {
        thetaX = 90;
        thetaY = 0;
        rotateDirection = 0;
		rotateMatrix = rotate(thetaX, xAxis);
        gl.uniformMatrix4fv(rotateLoc, false, flatten(rotateMatrix));
    };

    document.getElementById( "bottomButton" ).onclick = function () {
        thetaX = -90;
        thetaY = 0;
        rotateDirection = 0;
		rotateMatrix = rotate(thetaX, xAxis);
        gl.uniformMatrix4fv(rotateLoc, false, flatten(rotateMatrix));
    };

    document.getElementById( "leftButton" ).onclick = function () {
        thetaX = 0;
        thetaY = 90;
        rotateDirection = 0;
		rotateMatrix = rotate(thetaY, yAxis);
        gl.uniformMatrix4fv(rotateLoc, false, flatten(rotateMatrix));
    };

    document.getElementById( "rightButton" ).onclick = function () {
        thetaX = 0;
        thetaY = -90;
        rotateDirection = 0;
		rotateMatrix = rotate(thetaY, yAxis);
        gl.uniformMatrix4fv(rotateLoc, false, flatten(rotateMatrix));
    };

    
    // Event listener for slide
    document.getElementById("slide").onchange = function() {
        scaleValue = event.srcElement.value/25;
        scaleMatrix = scalem(scaleValue, scaleValue, scaleValue);
        gl.uniformMatrix4fv(scaleLoc, false, flatten(scaleMatrix));
    };

    // Event listener for mouse input
    canvas.addEventListener("click", function() {
        trans_X = -1 + 2*event.clientX/canvas.width;
        trans_Y = -1 + 2*(canvas.height - event.clientY)/canvas.height;
    });

    // Initial render commands
    gl.uniformMatrix4fv(translateLoc, false, flatten(translateMatrix));
    gl.uniformMatrix4fv(rotateLoc, false, flatten(rotateMatrix));
    gl.uniformMatrix4fv(scaleLoc, false, flatten(scaleMatrix));

    render();	
}

function withinEpsilon(x, y) {
    return x < y + 0.05 && x > y - 0.01;
}

// TODO: Edit this function.
function render()
{
    gl.clear( gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Animated Rotation
    switch(rotateDirection) {
        case 1:
            thetaX -= 1; break;
        case 2:
            thetaX += 1; break;
        case 3:
            thetaY -= 1; break;
        case 4:
            thetaY += 1; break;
        default:
            break;
    }
    rotateMatrix = rotate(thetaX, [1, 0, 0]);
    rotateMatrix = mult(rotateMatrix, rotate(thetaY, [0, 1, 0]));
    gl.uniformMatrix4fv(rotateLoc, false, flatten(rotateMatrix));

    // Animated Translation
    if(!withinEpsilon(current_X, trans_X)) current_X -= 0.05 * (current_X-trans_X)/Math.abs(current_X-trans_X);
    else current_X = trans_X;
    if(!withinEpsilon(current_Y, trans_Y)) current_Y -= 0.05 * (current_Y-trans_Y)/Math.abs(current_Y-trans_Y);
    else current_Y = trans_Y;
    translateMatrix = translate([current_X, current_Y, 0]);
    gl.uniformMatrix4fv(translateLoc, false, flatten(translateMatrix));

    gl.drawElements( gl.TRIANGLES, bunny_indices.length, gl.UNSIGNED_SHORT, 0);

    requestAnimFrame( render );
}

