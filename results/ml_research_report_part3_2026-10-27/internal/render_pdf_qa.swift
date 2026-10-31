import Foundation
import PDFKit
import AppKit
let root=CommandLine.arguments[1]
let doc=PDFDocument(url:URL(fileURLWithPath:root+"/report.pdf"))!
let dest=root+"/internal/qa"
try FileManager.default.createDirectory(atPath:dest,withIntermediateDirectories:true)
var rows:[[String:Any]]=[]
var thumbs:[NSImage]=[]
for i in 0..<doc.pageCount {
 let p=doc.page(at:i)!, box=p.bounds(for:.mediaBox)
 let pic=p.thumbnail(of:NSSize(width:1200,height:1698),for:.mediaBox)
 let data=NSBitmapImageRep(data:pic.tiffRepresentation!)!.representation(using:.png,properties:[:])!
 try data.write(to:URL(fileURLWithPath:dest+String(format:"/page_%02d.png",i+1)))
 thumbs.append(pic)
 let lines=(p.string ?? "").components(separatedBy:"\n")
 var outside=0
 for k in 0..<p.numberOfCharacters {
  let b=p.characterBounds(at:k)
  if b.width>0 && b.height>0 && (b.minX<15 || b.maxX>box.width-15 || b.minY<8 || b.maxY>box.height-8) {outside+=1}
 }
 rows.append(["page":i+1,"width":box.width,"height":box.height,"characters":p.numberOfCharacters,"outside_safe_page":outside,"first_lines":Array(lines.prefix(3)),"last_lines":Array(lines.suffix(2))])
}
for start in stride(from:0,to:thumbs.count,by:9) {
 let canvas=NSImage(size:NSSize(width:900,height:1314));canvas.lockFocus()
 NSColor.gray.setFill();NSRect(x:0,y:0,width:900,height:1314).fill()
 for j in 0..<min(9,thumbs.count-start) {
  let x=Double(j%3)*300.0,y=Double(2-j/3)*438.0
  thumbs[start+j].draw(in:NSRect(x:x+3,y:y+16,width:294,height:416))
  NSString(string:"Стр. \(start+j+1)").draw(at:NSPoint(x:x+8,y:y+1),withAttributes:[.font:NSFont.systemFont(ofSize:11),.foregroundColor:NSColor.black])
 }
 canvas.unlockFocus()
 let png=NSBitmapImageRep(data:canvas.tiffRepresentation!)!.representation(using:.png,properties:[:])!
 try png.write(to:URL(fileURLWithPath:dest+"/contact_\(start/9+1).png"))
}
let data=try JSONSerialization.data(withJSONObject:["pages":doc.pageCount,"page_checks":rows],options:[.prettyPrinted,.sortedKeys])
try data.write(to:URL(fileURLWithPath:root+"/internal/pdf_qa.json"))
print("Rendered \(doc.pageCount) pages and six contact sheets")
