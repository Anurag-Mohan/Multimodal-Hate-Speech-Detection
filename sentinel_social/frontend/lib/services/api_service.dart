import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  // 🔴 CRITICAL: PASTE YOUR NGROK URL HERE BEFORE BUILDING THE APK!
  // Example: 'https://1234-56-78-90.ngrok-free.app'
  // Do NOT put a trailing slash '/' at the end.
  static const String baseUrl =
      'https://crowded-bitterly-mafalda.ngrok-free.dev';

  static Future<Map<String, dynamic>> scanMedia(File imageFile) async {
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/api/scan'),
      );

      request.files.add(
        await http.MultipartFile.fromPath('file', imageFile.path),
      );

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        throw Exception('Failed to scan media: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error communicating with backend: $e');
    }
  }
}
