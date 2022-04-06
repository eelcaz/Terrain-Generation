// Obj Loader from K3D.js (http://k3d.ivank.net/?p=documentation)

function readLine(a, off)	// Uint8Array, offset
{
	var s = "";
	while(a[off] != 10) s += String.fromCharCode(a[off++]);
	return s;
}

function loadOBJFromPath(path, resp, _callback)
{
	var request = new XMLHttpRequest();
	request.open("GET", path, true);
	request.responseType = "arraybuffer";
	request.onload = function(e){resp(e.target.response, _callback);};
	request.send();
}

function loadOBJFromBuffer(buff)
{

	var res = {};
	res.groups = {};
	
	res.c_verts = [];
	res.c_uvt	= [];
	res.c_norms = [];
	
	res.i_verts = [];
	res.i_uvt   = [];
	res.i_norms = [];
	
	var cg = {from: 0, to:0};	// current group
	var off = 0;
	var a = new Uint8Array(buff);
	
	while(off < a.length)
	{
		var line = readLine(a, off);
		off += line.length + 1;
		line = line.replace(/ +(?= )/g,'');
		line = line.replace(/(^\s+|\s+$)/g, '');
		var cds = line.split(" ");
		if(cds[0] == "g")
		{
			cg.to = res.i_verts.length;
			if(res.groups[cds[1]] == null) res.groups[cds[1]] = {from:res.i_verts.length, to:0};
			cg = res.groups[cds[1]];
		}
		if(cds[0] == "v")
		{
			var x = parseFloat(cds[1]);
			var y = parseFloat(cds[2]);
			var z = parseFloat(cds[3]);
			res.c_verts.push(x,y,z);
		}
		if(cds[0] == "vt")
		{
			var x = parseFloat(cds[1]);
			var y = 1-parseFloat(cds[2]);
			res.c_uvt.push(x,y);
		}
		if(cds[0] == "vn")
		{
			var x = parseFloat(cds[1]);
			var y = parseFloat(cds[2]);
			var z = parseFloat(cds[3]);
			res.c_norms.push(x,y,z);
		}
		if(cds[0] == "f")
		{
			var v0a = cds[1].split("/"), v1a = cds[2].split("/"), v2a = cds[3].split("/");
			var vi0 = parseInt(v0a[0])-1, vi1 = parseInt(v1a[0])-1, vi2 = parseInt(v2a[0])-1;
			var ui0 = parseInt(v0a[1])-1, ui1 = parseInt(v1a[1])-1, ui2 = parseInt(v2a[1])-1;
			var ni0 = parseInt(v0a[2])-1, ni1 = parseInt(v1a[2])-1, ni2 = parseInt(v2a[2])-1;
			
			var vlen = res.c_verts.length/3, ulen = res.c_uvt.length/2, nlen = res.c_norms.length/3;
			if(vi0<0) vi0 = vlen + vi0+1; if(vi1<0) vi1 = vlen + vi1+1;	if(vi2<0) vi2 = vlen + vi2+1;
			if(ui0<0) ui0 = ulen + ui0+1; if(ui1<0) ui1 = ulen + ui1+1;	if(ui2<0) ui2 = ulen + ui2+1;
			if(ni0<0) ni0 = nlen + ni0+1; if(ni1<0) ni1 = nlen + ni1+1;	if(ni2<0) ni2 = nlen + ni2+1;
			
			res.i_verts.push(vi0, vi1, vi2);  //cg.i_verts.push(vi0, vi1, vi2)
			res.i_uvt  .push(ui0, ui1, ui2);  //cg.i_uvt  .push(ui0, ui1, ui2);
			res.i_norms.push(ni0, ni1, ni2);  //cg.i_norms.push(ni0, ni1, ni2);
			if(cds.length == 5)
			{
				var v3a = cds[4].split("/");
				var vi3 = parseInt(v3a[0])-1, ui3 = parseInt(v3a[1])-1, ni3 = parseInt(v3a[2])-1;
				if(vi3<0) vi3 = vlen + vi3+1;
				if(ui3<0) ui3 = ulen + ui3+1;
				if(ni3<0) ni3 = nlen + ni3+1;
				res.i_verts.push(vi0, vi2, vi3);  //cg.i_verts.push(vi0, vi2, vi3);
				res.i_uvt  .push(ui0, ui2, ui3);  //cg.i_uvt  .push(ui0, ui2, ui3);
				res.i_norms.push(ni0, ni2, ni3);  //cg.i_norms.push(ni0, ni2, ni3);
			}
		}
	}
	cg.to = res.i_verts.length;
	
	return res;
}